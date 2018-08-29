'''Collection of losses used to train our models.'''
import tensorflow as tf


def latent_kl(q, p):
    '''Kullback-Leible divergence on two embeddings/probabiblity distributions
    q and p.'''

    mean1 = q
    mean2 = p

    kl = 0.5 * tf.square(mean2 - mean1)
    kl = tf.reduce_sum(kl, axis = [1,2,3])
    kl = tf.reduce_mean(kl)
    return kl

def kl_loss(qs, ps):
    '''Sum over all :func:`latent_kl` losses for each ``q`` and ``p`` in ``qs``
    and ``ps``.'''

    total_kl_loss = 0.
    for q, p in zip(qs, ps):
        total_kl_loss += latent_kl(q, p)

    return total_kl_loss


def likelihood_loss(x, params, m, vgg19):
    '''Likelihood loss.

    Args:
        x (tf.Tensor): Original image.
        params (tf.Tensor): Reconstructed image.
        m (tf.Tensor): Original mask.
        vgg19 (VGG19Features): VGG19Features instance calculating the 
            deep feature loss.

    Returns:
        tf.op: Image reconstruction loss.
        tf.op: Mask reconstruction loss.
    '''
    rec_x, rec_m = tf.split(params, [3,1], axis = 3)
    x = x*m
    #rec_x = rec_x*m # both possible but lets predict masked image

    l = tf.to_float(0.0)
    l_rec = 5.0*vgg19.make_loss_op(x, rec_x)
    # rescale m to -1 1 to utilize full range
    m = 2.0*m -1.0
    l_mask = 5.0*tf.reduce_mean(tf.abs(m - rec_m))
    return l_rec, l_mask


def combined_loss(self,
                  weights,
                  original_image,
                  original_mask,
                  generated_params,
                  pose_embeddings,
                  appearance_embeddings,
                  vgg19):
    '''
    Args:
        weights (list or dict): Weights for each loss.
        original_image: See name
        original_mask: See name
        generated_params: Generated image and cnn features
        pose_embeddings: See name
        appearance_embeddings: See name
        vgg19 (VGG19Features): VGG19Features instance calculating the 
            deep feature loss.
    '''

    if isinstance(weights, dict):
        losses = list(weights.keys())
    else:
        losses = ['kl', 'rec']
        assert len(weights) == len(losses), 'Only use these losses: ' + losses
        weights = {l: w for l, w in zip(losses, weights)}

    total_loss = 0.

    for loss in losses:
        if loss == 'kl':
            amount = latent_kl(pose_embeddings, appearance_embeddings)
        elif loss == 'rec':
            amount = likelihood_loss(original_image,
                                     reconstruction_params,
                                     original_mask,
                                     vgg19)
        
        total_loss += weights[loss] * amount

    return combined_loss
