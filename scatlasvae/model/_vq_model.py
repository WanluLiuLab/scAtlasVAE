from ._gex_model import *
import einops

class scAtlasQVAE(scAtlasVAE):
    def __init__(self, *args, **kwargs):
        super(scAtlasQVAE, self).__init__(*args, **kwargs)
        self.quantitizer = nn.Embedding(self.n_label, self.n_latent).to(self.device)

    def encode(
        self, 
        X: torch.Tensor, 
        batch_index: torch.Tensor = None, 
        eps: float = 1e-4
    ):
        libsize = torch.log(X.sum(1))
        if self.reconstruction_method == 'zinb' or self.reconstruction_method == 'nb':
            if self.total_variational:
                X = self._normalize_data(X, after=1e4, copy=True)
            if self.log_variational:
                X = torch.log(1+X)
        q = self.encoder.encode(torch.hstack([X,libsize.unsqueeze(1)])) \
            if self.encode_libsize \
            else self.encoder.encode(X)
        q_mu = self.z_mean_fc(q)
        q_var = torch.exp(self.z_var_fc(q)) + eps
        z = Normal(q_mu, q_var.sqrt()).rsample()
        prediction = self.fc(z)
        z_vq = self.quantitizer(torch.argmax(prediction, dim=1))

        H = dict(
            q = q,
            q_mu = q_mu,
            q_var = q_var,
            z = z_vq,
            z_old = z,
            prediction = prediction
        )

        return H

    def forward(
        self,
        X: torch.Tensor,
        lib_size: torch.Tensor,
        batch_index: torch.Tensor = None,
        label_index: torch.Tensor = None,
        additional_label_index: torch.Tensor = None,
        additional_batch_index: torch.Tensor = None,
        P: torch.Tensor = None,
        reduction: str = "sum",
        compute_mmd: bool = False
    ):
        H = self.encode(X, batch_index)
        q_mu = H["q_mu"]
        q_var = H["q_var"]
        mean = torch.zeros_like(q_mu)
        scale = torch.ones_like(q_var)
        kldiv_loss = kld(Normal(q_mu, q_var.sqrt()), Normal(mean, scale)).sum(dim = 1)
        prediction_loss = torch.tensor(0., device=self.device)
        additional_prediction_loss = torch.tensor(0., device=self.device)
        R = self.decode(H, lib_size, batch_index, label_index, additional_batch_index)

        if self.reconstruction_method == 'zinb':
            reconstruction_loss = LossFunction.zinb_reconstruction_loss(
                X,
                mu = R['px_rna_scale'],
                theta = R['px_rna_rate'].exp(),
                gate_logits = R['px_rna_dropout'],
                reduction = reduction
            )
        elif self.reconstruction_method == 'nb':
            reconstruction_loss = LossFunction.nb_reconstruction_loss(
                X,
                mu = R['px_rna_scale'],
                theta = R['px_rna_rate'].exp(),
                reduction = reduction
            )
        elif self.reconstruction_method == 'zg':
            reconstruction_loss = LossFunction.zi_gaussian_reconstruction_loss(
                X,
                mean=R['px_rna_scale'],
                variance=R['px_rna_rate'].exp(),
                gate_logits=R['px_rna_dropout'],
                reduction=reduction
            )
        elif self.reconstruction_method == 'mse':
            X_norm = self._normalize_data(X, after=1e4)
            X_norm = torch.log(X_norm + 1)
            reconstruction_loss = nn.functional.mse_loss(
                X_norm,
                R['px_rna_scale'],
                reduction=reduction
            )
        else:
            raise ValueError(f"reconstruction_method {self.reconstruction_method} is not supported")

        if self.n_label > 0:
            criterion = nn.CrossEntropyLoss(weight=self.label_category_weight)

            prediction = H['prediction']

            if self.new_adata_code and self.new_adata_code in label_index:
                prediction_index = (label_index != self.new_adata_code).squeeze()
                prediction_loss = criterion(prediction[prediction_index], one_hot(label_index[prediction_index], self.n_label))
            else:
                prediction_loss = criterion(prediction, one_hot(label_index, self.n_label))

        if self.n_additional_label is not None:
            prediction_loss = prediction_loss * self.additional_label_weight[0]
            for e,i in enumerate(self.n_additional_label):
                criterion = nn.CrossEntropyLoss(weight=self.additional_label_category_weight[e])
                additional_prediction = self.additional_fc[e](H['z'])
                if self.additional_new_adata_code[e] and self.additional_new_adata_code[e] in additional_label_index[e]:
                    additional_prediction_index = (additional_label_index[e] != self.additional_new_adata_code[e]).squeeze()

                    additional_prediction_loss += criterion(
                        additional_prediction[additional_prediction_index],
                        one_hot(additional_label_index[e][additional_prediction_index], i) * self.additional_label_weight[e+1]
                    )
                else:
                    additional_prediction_loss += criterion(additional_prediction, one_hot(additional_label_index[e], i)) * self.additional_label_weight[e+1]

        latent_constrain = torch.tensor(0.)
        if self.constrain_latent_embedding and P is not None:
            # Constrains on cells with no PCA information will be ignored
            latent_constrain_mask = P.mean(1) != 0
            if self.constrain_latent_method == 'mse':
                latent_constrain = (
                    nn.MSELoss(reduction='none')(P, q_mu).sum(1) * latent_constrain_mask
                ).sum() / len(list(filter(lambda x: x != 0, P.detach().cpu().numpy().mean(1))))
            elif self.constrain_latent_method == 'normal':
                latent_constrain = (
                    kld(Normal(q_mu, q_var.sqrt()), Normal(P, torch.ones_like(P))).sum(1) * latent_constrain_mask
                ).sum() / len(list(filter(lambda x: x != 0, P.detach().cpu().numpy().mean(1))))

        mmd_loss = torch.tensor(0.)
        if self.mmd_key is not None and compute_mmd:
            if self.mmd_key == 'batch':
                mmd_loss = self.mmd_loss(
                    H['q_mu'],
                    batch_index.detach().cpu().numpy(),
                    dim=1
                )
            elif self.mmd_key == 'additional_batch':
                for i in range(len(self.additional_batch_keys)):
                    mmd_loss += self.mmd_loss(
                        H['q_mu'], 
                        additional_batch_index[i].detach().cpu().numpy(),
                        dim=1
                    )
            elif self.mmd_key == 'both':
                mmd_loss = self.mmd_loss(
                    H['q_mu'], 
                    batch_index.detach().cpu().numpy(),
                    dim=1
                )
                for i in range(len(self.additional_batch_keys)):
                    mmd_loss += self.hierarchical_mmd_loss_2(
                        H['q_mu'], 
                        batch_index.detach().cpu().numpy(),
                        additional_batch_index[i].detach().cpu().numpy(),
                        dim=1
                    )

        loss_record = {
            "reconstruction_loss": reconstruction_loss,
            "prediction_loss": prediction_loss,
            "additional_prediction_loss": additional_prediction_loss,
            "kldiv_loss": kldiv_loss,
            "mmd_loss": mmd_loss,
            "latent_constrain_loss": latent_constrain
        }
        return H, R, loss_record