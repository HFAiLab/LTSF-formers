import ml_collections as mlc

args = mlc.ConfigDict(
    {
        "model": "informer",        # model of experiment, options: [informer, informerstack, informerlight(TBD)]
        "data": "ETTh1",
        "root_path": "./data/ETT/",
        "data_path": "ETTh1.csv",
        "features": "M",            # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
        "target": "OT",             # target feature in S or MS task
        "freq": "h",                # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
        "checkpoints": "./checkpoints/",
        "seed": 42,

        "seq_len": 48,
        "label_len": 48,
        "pred_len": 24,

        "enc_in": 7,                # encoder input size
        "dec_in": 7,                # decoder input size
        "c_out": 7,                 # output size
        "d_model": 512,             # dimension of model
        "n_heads": 8,               # num of heads
        "e_layers": 2,              # num of encoder layers
        "d_layers": 1,              # num of decoder layers
        "s_layers": [3,2,1],        # num of stack encoder layers
        "d_ff": 2048,               # dimension of fcn
        "factor": 5,                # probsparse attn factor
        "padding": 0,               # padding type
        "distil": True,             # whether to use distilling in encoder, using this argument means not using distilling
        "dropout": 0.05,            # dropout
        "attn": 'prob',             # attention used in encoder, options:[prob, full]
        "embed": 'timeF',           # time features encoding, options:[timeF, fixed, learned]
        "activation": 'gelu',       # activation
        "output_attention": False,  # whether to output attention in ecoder
        "do_predict": False,        # whether to predict unseen future data
        "mix": True,                # use mix attention in generative decoder
        "itr": 10,                  # experiments times
        "train_epochs": 6,          # train epochs
        "batch_size": 32,           # batch size of train input data
        "patience": 3,              # early stopping patience
        "learning_rate": 0.0001,    # optimizer learning rate
        "des": 'Exp',               # exp description
        "loss": 'mse',              # loss function
        "lradj": 'type1',           # adjust learning rate
        "use_amp": False,           # use automatic mixed precision training
        "inverse": False,           # inverse output data
    }
)
