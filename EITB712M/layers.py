from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None

    dim_size = x[0].shape
    X = x.reshape(x.shape[0], np.prod(dim_size))
    out = X.dot(w) + b

    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None

    dim_shape = np.prod(x[0].shape)
    N = x.shape[0]
    X = x.reshape(N, dim_shape)
    # input gradient
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)
    # weight gradient
    dw = X.T.dot(dout)
    # bias gradient
    db = dout.sum(axis=0)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None

    out = np.maximum(0, x)

    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache

    dx = dout * (x > 0)

    return dx



def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HF and width WF.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HF, WF)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HF) / stride
      W' = 1 + (W + 2 * pad - WF) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: You can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Parameter und Dimensionen holen
    P = conv_param['pad']
    S = conv_param['stride']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape #? Hier stand noch ein C, aber das ist ja schon in x.shape enthalten (C wird aber nicht genutzt)

    # Output-Dimensionen berechnen
    H_out = 1 + (H + 2 * P - HH) // S
    W_out = 1 + (W + 2 * P - WW) // S

    # Input mit Nullen padden
    # np.pad-Format: ((dim0_before, dim0_after), (dim1_before, dim1_after), ...)
    x_pad = np.pad(x, ((0, 0), (0, 0), (P, P), (P, P)), 'constant')

    # Output-Volumen initialisieren
    out = np.zeros((N, F, H_out, W_out))

    # Naive Implementierung mit 4 verschachtelten Schleifen
    for n in range(N):          # Für jedes Bild im Batch
        for f in range(F):      # Für jeden Filter
            for h_out in range(H_out): # Für jede Output-Zeile
                for w_out in range(W_out): # Für jede Output-Spalte
                    
                    # Start-Koordinaten des Patches im gepaddeten Input
                    h_start = h_out * S
                    w_start = w_out * S
                    
                    # Patch extrahieren (Shape: C, HH, WW)
                    x_patch = x_pad[n, :, h_start:h_start + HH, w_start:w_start + WW]
                    
                    # Filter extrahieren (Shape: C, HH, WW)
                    filter_w = w[f, :, :, :]
                    
                    # Faltung durchführen: Elementweise Multiplikation und Summe
                    # (C, HH, WW) * (C, HH, WW) -> Summe ergibt Skalar
                    conv_sum = np.sum(x_patch * filter_w)
                    
                    # Bias addieren und im Output speichern
                    out[n, f, h_out, w_out] = conv_sum + b[f]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, w, b, conv_param = cache
    P = conv_param['pad']
    S = conv_param['stride']
    
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_out, W_out = dout.shape

    # Gradienten-Matrizen initialisieren
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # Input padden (für dx- und dw-Berechnung)
    x_pad = np.pad(x, ((0, 0), (0, 0), (P, P), (P, P)), 'constant')
    # dx_pad initialisieren (hier sammeln wir die Gradienten)
    dx_pad = np.zeros_like(x_pad)

    # 1. db berechnen (Shape: F,)
    # Summiere dout über N, H_out, W_out
    db = np.sum(dout, axis=(0, 2, 3))

    # 2. & 3. dx und dw berechnen
    for n in range(N):          # Für jedes Bild
        for f in range(F):      # Für jeden Filter
            for h_out in range(H_out): # Für jede Output-Zeile
                for w_out in range(W_out): # Für jede Output-Spalte
                    
                    # Start-Koordinaten
                    h_start = h_out * S
                    w_start = w_out * S
                    
                    # Upstream-Gradient (Skalar)
                    d_out_scalar = dout[n, f, h_out, w_out]
                    
                    # Patch aus x_pad extrahieren (C, HH, WW)
                    x_patch = x_pad[n, :, h_start:h_start + HH, w_start:w_start + WW]
                    
                    # dw berechnen:
                    # Gradient für w[f] ist der x_patch * upstream_gradient
                    dw[f, :, :, :] += x_patch * d_out_scalar
                    
                    # dx_pad berechnen:
                    # Gradient für den x_patch ist w[f] * upstream_gradient
                    dx_pad[n, :, h_start:h_start + HH, w_start:w_start + WW] += w[f, :, :, :] * d_out_scalar

    # 4. dx aus dx_pad extrahieren (Padding entfernen)
    dx = dx_pad[:, :, P:P + H, P:P + W]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """

    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    PH = pool_param['pool_height']
    PW = pool_param['pool_width']
    S = pool_param['stride']

    # Output-Dimensionen
    H_out = 1 + (H - PH) // S
    W_out = 1 + (W - PW) // S

    out = np.zeros((N, C, H_out, W_out))

    for n in range(N):
        for c in range(C):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    
                    # Start-Koordinaten des Patches
                    h_start = h_out * S
                    w_start = w_out * S
                    
                    # Patch extrahieren
                    x_patch = x[n, c, h_start:h_start + PH, w_start:w_start + PW]
                    
                    # Max-Wert finden
                    out[n, c, h_out, w_out] = np.max(x_patch)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache
    N, C, H, W = x.shape
    PH = pool_param['pool_height']
    PW = pool_param['pool_width']
    S = pool_param['stride']
    _, _, H_out, W_out = dout.shape

    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    
                    # Start-Koordinaten
                    h_start = h_out * S
                    w_start = w_out * S
                    
                    # Upstream-Gradient (Skalar)
                    d_out_scalar = dout[n, c, h_out, w_out]
                    
                    # Patch extrahieren
                    x_patch = x[n, c, h_start:h_start + PH, w_start:w_start + PW]
                    
                    # Den flachen Index des Max-Werts finden
                    # (z.B. in einem 2x2-Patch ist das ein Index von 0 bis 3)
                    max_idx_flat = np.argmax(x_patch)
                    
                    # Den flachen Index zurück in 2D-Koordinaten (h, w) umwandeln
                    max_idx_h, max_idx_w = np.unravel_index(max_idx_flat, (PH, PW))
                    
                    # Den Gradienten nur an diese eine (maximale) Position im dx-Array addieren
                    dx[n, c, h_start + max_idx_h, w_start + max_idx_w] += d_out_scalar

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx



def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
