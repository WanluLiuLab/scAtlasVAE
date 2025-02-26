import torch
from torch import nn

from ._gex_model import (
    scAtlasVAE,
    LossFunction,
    one_hot
)
from ._gex_model import *
import torch.nn.functional as F

class scAtlasVQVAE(scAtlasVAE):
    def __init__(self, *args, **kwargs):

        super(scAtlasVQVAE, self).__init__(*args, **kwargs)

        self.quantitizer = nn.Embedding(
            self.n_label, 
            self.n_latent,
        ).to(self.device)
        self.quantitizer.weight.data.uniform_(0,0)

        self.to(self.device)

        pretrained_state_dict = kwargs.get("pretrained_state_dict", None)
        if pretrained_state_dict is not None:
            if isinstance(pretrained_state_dict, str):
                pretrained_state_dict = torch.load(pretrained_state_dict)['model_state_dict']
            self.partial_load_state_dict(pretrained_state_dict)

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

        prediction = self.fc(q_mu)

        z_vq = (self.quantitizer.weight * F.softmax(prediction, dim=1).unsqueeze(-1)).sum(1)

        H = dict(
            q = q,
            q_mu=q_mu,
            q_var=q_var,
            z = z,
            z_vq = z_vq,
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

        R = self.decode(
            H, 
            lib_size, 
            batch_index, 
            label_index, 
            additional_batch_index
        )

        # TODO: check if this is ok: use dummy batch variables for decoding 
        r = self.decode(
            {"z": H["z_vq"]},
            lib_size,
            torch.ones(batch_index.shape, device=self.device) * self.n_batch,
            label_index,
            [
                torch.ones(additional_batch_index[e].shape, device=self.device)
                * self.n_additional_batch[e]
                for e in range(len(additional_batch_index))
            ],
        )

        reconstruction_loss = LossFunction.zinb_reconstruction_loss(
                X,
                mu = R['px_rna_scale'],
                theta = R['px_rna_rate'].exp(),
                gate_logits = R['px_rna_dropout'],
                reduction = reduction
        )
        reconstruction_vq_loss = LossFunction.zinb_reconstruction_loss(
                X,
                mu = r['px_rna_scale'],
                theta = r['px_rna_rate'].exp(),
                gate_logits = r['px_rna_dropout'],
                reduction = reduction
        )
        vq_loss = torch.mean((H['z_vq'].detach()-H['z'])**2) + \
            torch.mean((H['z_vq'] - H['z'].detach()) ** 2)

        if self.n_label > 0:
            criterion = nn.CrossEntropyLoss(weight=self.label_category_weight)

            prediction = self.fc(q_mu)

            if self.new_adata_code and self.new_adata_code in label_index:
                prediction_index = (label_index != self.new_adata_code).squeeze()
                prediction_loss = criterion(
                    prediction[prediction_index],
                    one_hot(label_index[prediction_index], self.n_label),
                )
            else:
                prediction_loss = criterion(
                    prediction, one_hot(label_index, self.n_label)
                )

        if self.n_additional_label is not None:
            prediction_loss = prediction_loss * self.additional_label_weight[0]
            for e, i in enumerate(self.n_additional_label):
                criterion = nn.CrossEntropyLoss(
                    weight=self.additional_label_category_weight[e]
                )
                additional_prediction = H['additional_prediction'][e]
                if (
                    self.additional_new_adata_code[e]
                    and self.additional_new_adata_code[e] in additional_label_index[e]
                ):
                    additional_prediction_index = (
                        additional_label_index[e] != self.additional_new_adata_code[e]
                    ).squeeze()

                    additional_prediction_loss += criterion(
                        additional_prediction[additional_prediction_index],
                        one_hot(
                            additional_label_index[e][additional_prediction_index], i
                        )
                        * self.additional_label_weight[e + 1],
                    )
                else:
                    additional_prediction_loss += (
                        criterion(
                            additional_prediction, one_hot(additional_label_index[e], i)
                        )
                        * self.additional_label_weight[e + 1]
                    )

        loss_record = {
            "reconstruction_loss": reconstruction_loss,
            "reconstruction_vq_loss": reconstruction_vq_loss,
            "prediction_loss": prediction_loss,
            "additional_prediction_loss": additional_prediction_loss,
            "kldiv_loss": kldiv_loss,
            "mmd_loss": torch.tensor(0.),
            "latent_constrain_loss": torch.tensor(0.),
            "vq_loss": vq_loss

        }
        return H, R, loss_record

    def fit(self,
        max_epoch: Optional[int] = None,
        n_per_batch:int = 128,
        kl_weight: float = 5.,
        pred_weight: float = 5.,
        mmd_weight: float = 1.,
        vq_weight: float = 1e2,
        vq_last_n_epoch: int = 10,
        gate_weight: float = 1.,
        constrain_weight: float = 1.,
        optimizer_parameters: Iterable = None,
        validation_split: float = .2,
        lr: bool = 5e-5,
        lr_schedule: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_min: float = 1e-6,
        n_epochs_kl_warmup: Union[int, None] = 400,
        weight_decay: float = 1e-6,
        random_seed: int = 12,
        subset_indices: Union[torch.tensor, np.ndarray] = None,
        pred_last_n_epoch: int = 10,
        pred_last_n_epoch_fconly: bool = False,
        compute_batch_after_n_epoch: int = 0,
        reconstruction_reduction: str = 'sum',
        n_concurrent_batch: int = 1,
    ):
        """
        Fit the model.
        
        :param max_epoch: int. Maximum number of epoch to train the model. If not provided, the model will be trained for 400 epochs or 20000 / n_record * 400 epochs as default.
        :param n_per_batch: int. Number of cells per batch.
        :param kl_weight: float. (Maximum) weight of the KL divergence loss.
        :param pred_weight: float. weight of the prediction loss.
        :param mmd_weight: float. weight of the mmd loss. ignored if mmd_key is None
        :param constrain_weight: float. weight of the constrain loss. ignored if constrain_latent_embedding is False. 
        :param optimizer_parameters: Iterable. Parameters to be optimized. If not provided, all parameters will be optimized.
        :param validation_split: float. Percentage of data to be used as validation set.
        :param lr: float. Learning rate.
        :param lr_schedule: bool. Whether to use learning rate scheduler.
        :param lr_factor: float. Factor to reduce learning rate.
        :param lr_patience: int. Number of epoch to wait before reducing learning rate.
        :param lr_threshold: float. Threshold to trigger learning rate reduction.
        :param lr_min: float. Minimum learning rate.
        :param n_epochs_kl_warmup: int. Number of epoch to warmup the KL divergence loss (deterministic warm-up
        of the KL-term).
        :param weight_decay: float. Weight decay (L2 penalty).
        :param random_seed: int. Random seed.
        :param subset_indices: Union[torch.tensor, np.ndarray]. Indices of cells to be used for training. If not provided, all cells will be used.
        :param pred_last_n_epoch: int. Number of epoch to train the prediction layer only.
        :param pred_last_n_epoch_fconly: bool. Whether to train the prediction layer only.
        :param reconstruction_reduction: str. Reduction method for reconstruction loss. Can be 'sum' or 'mean'.
        """
        self.train()
        if max_epoch is None:
            max_epoch = np.min([round((20000 / self._n_record ) * 400), 400])
            mt(f"max_epoch is not provided, setting max_epoch to {max_epoch}")
        if n_epochs_kl_warmup:
            n_epochs_kl_warmup = min(max_epoch, n_epochs_kl_warmup)
            kl_warmup_gradient = kl_weight / n_epochs_kl_warmup
            kl_weight_max = kl_weight
            kl_weight = 0.

        if optimizer_parameters is None:
            optimizer = optim.AdamW(
                chain(
                    *list(map(lambda z: getattr(self, z).parameters(), filter(lambda x: not x.startswith("quantitizer"), np.unique(list(map(lambda x: x.split(".")[0], self.state_dict().keys()))))))
                ),
                lr, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(optimizer_parameters, lr, weight_decay=weight_decay)

        vq_optimizer = optim.AdamW(
            self.quantitizer.parameters(),
            lr * 10,  weight_decay=weight_decay
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=lr_patience,
            factor=lr_factor,
            threshold=lr_threshold,
            min_lr=lr_min,
            threshold_mode="abs",
            verbose=True,
        ) if lr_schedule else None

        labels=None

        best_state_dict = None
        best_score = 0
        current_score = 0
        pbar = get_tqdm()(
            range(max_epoch), 
            desc="Epoch", 
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}' if not is_notebook() else '',
            position=0, 
            leave=True
        )
        loss_record = {
            "epoch_reconstruction_loss": 0,
            "epoch_kldiv_loss": 0,
            "epoch_prediction_loss": 0,
            "epoch_mmd_loss": 0,
            "epoch_total_loss": 0
        }

        epoch_total_loss_list = []
        epoch_reconstruction_loss_list = []
        epoch_reconstruction_vq_loss_list = []
        epoch_kldiv_loss_list = []
        epoch_prediction_loss_list = []
        epoch_mmd_loss_list = []
        epoch_gate_loss_list = []
        epoch_constraint_loss_list = []
        epoch_vq_loss_list = []

        for epoch in range(1, max_epoch+1):
            self._trained = True
            pbar.desc = "Epoch {}".format(epoch)
            epoch_total_loss = 0
            epoch_reconstruction_loss = 0
            epoch_reconstruction_vq_loss = 0
            epoch_kldiv_loss = 0
            epoch_prediction_loss = 0
            epoch_mmd_loss = 0
            epoch_gate_loss = 0
            epoch_constrain_loss = 0
            epoch_vq_loss = 0

            X_train, X_test = self.as_dataloader(
                n_per_batch=n_per_batch,
                train_test_split = True,
                validation_split = validation_split,
                random_seed=random_seed,
                subset_indices=subset_indices
            )

            if self.n_label > 0 and epoch == max_epoch - pred_last_n_epoch:
                mt("saving transcriptome only state dict")
                self.gene_only_state_dict = deepcopy(self.state_dict())
                if  pred_last_n_epoch_fconly:
                    optimizer = optim.AdamW(chain(self.att.parameters(), self.fc.parameters()), lr, weight_decay=weight_decay)

            X_train = list(X_train) # convert to list

            future_dict = {}

            with ThreadPoolExecutor(max_workers=1) as executor:
                for b, batch_indices in enumerate(X_train):
                    future = future_dict.get(b, None)
                    if future is None:
                        X, P, batch_index, label_index, additional_label_index, additional_batch_index, lib_size = self._prepare_batch(batch_indices)
                    else:
                        X, P, batch_index, label_index, additional_label_index, additional_batch_index, lib_size = future.result()
                        # future.clear()
                        future_dict.pop(b)

                    for fb in range(b+1, b+1+n_concurrent_batch):
                        if fb < len(X_train):
                            future_dict[fb] = executor.submit(self._prepare_batch, X_train[fb])

                    H, R, L = self.forward(
                        X,
                        lib_size,
                        batch_index,
                        label_index,
                        additional_label_index,
                        additional_batch_index,
                        P,
                        reduction=reconstruction_reduction,
                        compute_mmd = mmd_weight > 0 and epoch >= compute_batch_after_n_epoch
                    )

                    reconstruction_loss = L['reconstruction_loss']
                    reconstruction_vq_loss = L['reconstruction_vq_loss']
                    prediction_loss = pred_weight * L['prediction_loss']
                    additional_prediction_loss = pred_weight * L['additional_prediction_loss']
                    kldiv_loss = L['kldiv_loss']
                    mmd_loss = mmd_weight * L['mmd_loss']
                    vq_loss = L['vq_loss']

                    avg_gate_loss = gate_weight * torch.sigmoid(R['px_rna_dropout']).sum(dim=1).mean()

                    avg_reconstruction_loss = reconstruction_loss.sum()  / n_per_batch
                    avg_reconstruction_vq_loss = reconstruction_vq_loss.sum() / n_per_batch
                    avg_kldiv_loss = kldiv_loss.sum()  / n_per_batch
                    avg_mmd_loss = mmd_loss / n_per_batch

                    epoch_reconstruction_loss += avg_reconstruction_loss.item()
                    epoch_reconstruction_vq_loss += avg_reconstruction_vq_loss.item()
                    epoch_kldiv_loss += avg_kldiv_loss.item()
                    epoch_mmd_loss += avg_mmd_loss.item()
                    epoch_gate_loss += avg_gate_loss.item()
                    epoch_vq_loss += vq_loss.item()

                    if self.n_label > 0:
                        epoch_prediction_loss += prediction_loss.sum().item()

                    if epoch > max_epoch - pred_last_n_epoch:
                        loss = avg_reconstruction_loss + avg_kldiv_loss * kl_weight + avg_mmd_loss + (prediction_loss.sum() + additional_prediction_loss.sum()) / (len(self.n_additional_label) if self.n_additional_label is not None else 0 + 1) + avg_gate_loss
                    else:
                        loss = avg_reconstruction_loss + avg_kldiv_loss * kl_weight + avg_mmd_loss + avg_gate_loss

                    if epoch > max_epoch - vq_last_n_epoch:
                        vqloss = vq_loss * vq_weight + avg_reconstruction_vq_loss
                        vq_optimizer.zero_grad()
                        vqloss.backward(retain_graph=True)
                        vq_optimizer.step()

                    if self.constrain_latent_embedding:
                        loss += constrain_weight * L['latent_constrain_loss']
                        epoch_constrain_loss += L['latent_constrain_loss'].item()

                    epoch_total_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix({
                        'rec': '{:.2e}'.format(loss_record["epoch_reconstruction_loss"]),
                        'kl': '{:.2e}'.format(loss_record["epoch_kldiv_loss"]),
                        'pred': '{:.2e}'.format(loss_record["epoch_prediction_loss"]),
                        'step': f'{b} / {len(X_train)}'
                    })
            loss_record = self.calculate_metric(X_test, kl_weight, pred_weight, mmd_weight, reconstruction_reduction)
            if lr_schedule:
                scheduler.step(loss_record["epoch_total_loss"])
            pbar.set_postfix({
                'rec': '{:.2e}'.format(loss_record["epoch_reconstruction_loss"]),
                'kl': '{:.2e}'.format(loss_record["epoch_kldiv_loss"]),
                'pred': '{:.2e}'.format(loss_record["epoch_prediction_loss"]),
            })
            epoch_total_loss_list.append(epoch_total_loss)
            epoch_reconstruction_loss_list.append(epoch_reconstruction_loss)
            epoch_reconstruction_vq_loss_list.append(epoch_reconstruction_vq_loss)
            epoch_kldiv_loss_list.append(epoch_kldiv_loss)
            epoch_prediction_loss_list.append(epoch_prediction_loss)
            epoch_mmd_loss_list.append(epoch_mmd_loss)
            epoch_gate_loss_list.append(epoch_gate_loss)
            epoch_constraint_loss_list.append(epoch_constrain_loss)
            epoch_vq_loss_list.append(epoch_vq_loss)
            pbar.update(1)
            if n_epochs_kl_warmup:
                kl_weight = min( kl_weight + kl_warmup_gradient, kl_weight_max)
            random_seed += 1
        if current_score < best_score:
            mt("restoring state dict with best performance")
            self.load_state_dict(best_state_dict)
        pbar.close()
        self.trained_state_dict = deepcopy(self.state_dict())
        return dict(
            epoch_total_loss_list=epoch_total_loss_list,   
            epoch_reconstruction_loss_list=epoch_reconstruction_loss_list,
            epoch_reconstruction_vq_loss_list=epoch_reconstruction_vq_loss_list,
            epoch_kldiv_loss_list=epoch_kldiv_loss_list,
            epoch_prediction_loss_list=epoch_prediction_loss_list,
            epoch_mmd_loss_list=epoch_mmd_loss_list,
            epoch_gate_loss_list=epoch_gate_loss_list,
            epoch_constraint_loss_list=epoch_constraint_loss_list,
            epoch_vq_loss_list=epoch_vq_loss_list
        )
