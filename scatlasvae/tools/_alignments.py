from collections import Counter
import numpy as np
import scanpy as sc


def cell_type_alignment(
    adata: sc.AnnData,
    obs_1: str,
    obs_2: str,
    palette_1: dict = None,
    palette_2: dict = None,
    perc_in_obs_1: float = 0.1,
    perc_in_obs_2: float = 0.1,
    ignore_label: str = "undefined",
    return_fig: bool = True,
):
    """
    Plot a Sankey diagram of cell type alignment between two obs columns.

    :param adata: Annotated data matrix.
    :param obs_1: First obs column to compare.
    :param obs_2: Second obs column to compare.
    :param palette_1: Color palette for obs_1.
    :param palette_2: Color palette for obs_2.
    :param perc_in_obs_1: Minimum percentage of cells in obs_1 to be considered.

    :return: Sankey diagram in plotly.graph_objects.Figure format.

    :Example:
        >>> import scatlasvae
        >>> fig = scatlasvae.utils.cell_type_alignment(adata, "cell_type", "cell_type_2")
        >>> fig.show()
        >>> fig.write_html("cell_type_alignment.html")
    """
    if obs_1 not in adata.obs.columns:
        raise ValueError(f"obs_1 {obs_1} not in adata.obs.columns")
    if obs_2 not in adata.obs.columns:
        raise ValueError(f"obs_2 {obs_2} not in adata.obs.columns")
    if palette_1 is None:
        try:
            palette_1 = sc.pl._tools.scatterplots._get_palette(adata, obs_1)
        except:
            palette_1 = dict(
                zip(
                    np.unique(adata.obs[obs_1]),
                    ["#000000"] * len(np.unique(adata.obs[obs_1])),
                )
            )
    if palette_2 is None:
        try:
            palette_2 = sc.pl._tools.scatterplots._get_palette(adata, obs_2)
        except:
            palette_2 = dict(
                zip(
                    np.unique(adata.obs[obs_2]),
                    ["#000000"] * len(np.unique(adata.obs[obs_2])),
                )
            )
    count = {}

    c1 = Counter(adata.obs.loc[adata.obs[obs_1] != ignore_label, obs_1])
    c2 = Counter(adata.obs.loc[adata.obs[obs_2] != ignore_label, obs_2])


    agg = adata.obs.groupby(obs_1).agg({obs_2: Counter})
    for i, j in zip(agg.index, agg.iloc[:, 0]):
        for k, v in j.items():
            if i != ignore_label and k != ignore_label:
                count[(i, k)] = v

    count = dict(
        list(filter(lambda x: x[1] / c1[x[0][0]] > perc_in_obs_1 and \
           x[1] / c2[x[0][1]] > perc_in_obs_2 , count.items()))
    )
    if not return_fig:
        return count
    else:
        try:
            import plotly.graph_objects as go
        except:
            raise ImportError("Please install plotly to use this function.")

        labels = list(np.unique(adata.obs[obs_1])) + list(np.unique(adata.obs[obs_2]))
        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=labels,
                        color="blue",
                    ),
                    link=dict(
                        source=list(
                            map(
                                lambda x: labels.index(x),
                                list(map(lambda z: z[0], count.keys())),
                            )
                        ),  # indices correspond to labels, eg A1, A2, A1, B1, ...
                        target=list(
                            map(
                                lambda x: labels.index(x),
                                list(map(lambda z: z[1], count.keys())),
                            )
                        ),
                        value=list(count.values()),
                    ),
                )
            ]
        )

        return count, fig
