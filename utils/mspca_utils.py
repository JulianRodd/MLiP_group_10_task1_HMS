from mspca import mspca


def apply_mspca_to_signal(x, wavelet_func="db4", threshold=0.4):
    x = x.flatten()
    mymodel = mspca.MultiscalePCA()
    x_2d = x.reshape(-1, 1)

    x_transformed = mymodel.fit_transform(x_2d, wavelet_func, threshold)
    x_transformed = x_transformed.flatten()

    return x_transformed
