
import tensorflow as tf

def P_tensor(model):
    P = tf.concat([node.p_tensor() for node in model.nodes],0)
    dofs = tf.concat(model.DOF,0) - 1
    dofs = tf.where(tf.math.less(dofs,model.nf+1))
    dofs = tf.squeeze(dofs)
    print(dofs)
    P = tf.gather(P,dofs)
    return P