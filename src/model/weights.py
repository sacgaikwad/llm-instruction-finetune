import numpy as np
import torch


def assign(left, right):

    if left.shape != right.shape:

        raise ValueError(
            f"Shape mismatch. "
            f"Left: {left.shape}, "
            f"Right: {right.shape}"
        )

    return torch.nn.Parameter(
        torch.tensor(
            right,
            dtype=left.dtype
        )
    )


def load_weights_into_model(
    model,
    params
):

    # ========================================================
    # Embeddings
    # ========================================================

    model.pos_emb.weight = assign(
        model.pos_emb.weight,
        params["wpe"]
    )

    model.tok_emb.weight = assign(
        model.tok_emb.weight,
        params["wte"]
    )

    # ========================================================
    # Transformer Blocks
    # ========================================================

    for b in range(
        len(params["blocks"])
    ):

        block_params = params[
            "blocks"
        ][b]

        # ----------------------------------------------------
        # Attention weights
        # ----------------------------------------------------

        q_w, k_w, v_w = np.split(
            block_params[
                "attn"
            ][
                "c_attn"
            ][
                "w"
            ],
            3,
            axis=-1
        )

        model.trf_blocks[
            b
        ].attention.W_query.weight = assign(
            model.trf_blocks[
                b
            ].attention.W_query.weight,
            q_w.T
        )

        model.trf_blocks[
            b
        ].attention.W_key.weight = assign(
            model.trf_blocks[
                b
            ].attention.W_key.weight,
            k_w.T
        )

        model.trf_blocks[
            b
        ].attention.W_value.weight = assign(
            model.trf_blocks[
                b
            ].attention.W_value.weight,
            v_w.T
        )

        # ----------------------------------------------------
        # Attention biases
        # ----------------------------------------------------

        q_b, k_b, v_b = np.split(
            block_params[
                "attn"
            ][
                "c_attn"
            ][
                "b"
            ],
            3,
            axis=-1
        )

        model.trf_blocks[
            b
        ].attention.W_query.bias = assign(
            model.trf_blocks[
                b
            ].attention.W_query.bias,
            q_b
        )

        model.trf_blocks[
            b
        ].attention.W_key.bias = assign(
            model.trf_blocks[
                b
            ].attention.W_key.bias,
            k_b
        )

        model.trf_blocks[
            b
        ].attention.W_value.bias = assign(
            model.trf_blocks[
                b
            ].attention.W_value.bias,
            v_b
        )

        # ----------------------------------------------------
        # Attention output projection
        # ----------------------------------------------------

        model.trf_blocks[
            b
        ].attention.out_proj.weight = assign(
            model.trf_blocks[
                b
            ].attention.out_proj.weight,
            block_params[
                "attn"
            ][
                "c_proj"
            ][
                "w"
            ].T
        )

        model.trf_blocks[
            b
        ].attention.out_proj.bias = assign(
            model.trf_blocks[
                b
            ].attention.out_proj.bias,
            block_params[
                "attn"
            ][
                "c_proj"
            ][
                "b"
            ]
        )

        # ----------------------------------------------------
        # Feed-forward
        # ----------------------------------------------------

        model.trf_blocks[
            b
        ].feed_forward.layers[
            0
        ].weight = assign(
            model.trf_blocks[
                b
            ].feed_forward.layers[
                0
            ].weight,
            block_params[
                "mlp"
            ][
                "c_fc"
            ][
                "w"
            ].T
        )

        model.trf_blocks[
            b
        ].feed_forward.layers[
            0
        ].bias = assign(
            model.trf_blocks[
                b
            ].feed_forward.layers[
                0
            ].bias,
            block_params[
                "mlp"
            ][
                "c_fc"
            ][
                "b"
            ]
        )

        model.trf_blocks[
            b
        ].feed_forward.layers[
            2
        ].weight = assign(
            model.trf_blocks[
                b
            ].feed_forward.layers[
                2
            ].weight,
            block_params[
                "mlp"
            ][
                "c_proj"
            ][
                "w"
            ].T
        )

        model.trf_blocks[
            b
        ].feed_forward.layers[
            2
        ].bias = assign(
            model.trf_blocks[
                b
            ].feed_forward.layers[
                2
            ].bias,
            block_params[
                "mlp"
            ][
                "c_proj"
            ][
                "b"
            ]
        )

        # ----------------------------------------------------
        # Layer normalization
        # ----------------------------------------------------

        model.trf_blocks[
            b
        ].norm1.scale = assign(
            model.trf_blocks[
                b
            ].norm1.scale,
            block_params[
                "ln_1"
            ][
                "g"
            ]
        )

        model.trf_blocks[
            b
        ].norm1.shift = assign(
            model.trf_blocks[
                b
            ].norm1.shift,
            block_params[
                "ln_1"
            ][
                "b"
            ]
        )

        model.trf_blocks[
            b
        ].norm2.scale = assign(
            model.trf_blocks[
                b
            ].norm2.scale,
            block_params[
                "ln_2"
            ][
                "g"
            ]
        )

        model.trf_blocks[
            b
        ].norm2.shift = assign(
            model.trf_blocks[
                b
            ].norm2.shift,
            block_params[
                "ln_2"
            ][
                "b"
            ]
        )

    # ========================================================
    # Final Layer Normalization
    # ========================================================

    model.final_norm.scale = assign(
        model.final_norm.scale,
        params["g"]
    )

    model.final_norm.shift = assign(
        model.final_norm.shift,
        params["b"]
    )

    # ========================================================
    # GPT-2 Output Head
    # ========================================================

    model.out_head.weight = assign(
        model.out_head.weight,
        params["wte"]
    )