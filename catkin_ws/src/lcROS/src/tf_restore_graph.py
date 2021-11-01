
import tensorflow as tf
from tensorflow.python import ops
import random

class RestoredVariable(tf.Variable):
    """
    A variable restored from disk
    """
    def __init__(self, name, trainable=True, collections=None, graph=None):
        if graph is None:
            graph = tf.get_default_graph()

        if collections is None:
            collections = [ops.GraphKeys.VARIABLES]
        if trainable and ops.GraphKeys.TRAINABLE_VARIABLES not in collections:
            # pylint: disable=g-no-augmented-assignment
            #
            # Pylint wants us to write collections += [...TRAINABLE_VARIABLES] which
            # is not the same (it modifies the list in place.)  Here, we only want to
            # modify the value of the variable, not the list.
            collections = collections + [ops.GraphKeys.TRAINABLE_VARIABLES]
            # pylint: enable=g-no-augmented-assignment
        
        self._variable = graph.as_graph_element(name).outputs[0]
        self._snapshot = graph.as_graph_element(name + '/read').outputs[0]
        self._initializer_op = graph.as_graph_element(name + '/Assign')
        
        i_name = name + '/Initializer/'
        keys = [k for k in graph._nodes_by_name.keys() if k.startswith(i_name) and '/' not in k[len(i_name):] ]
        if len(keys) != 1:
            raise ValueError('Could not find initializer for variable', keys)
        
        self._initial_value = None #initial_value node

        for key in collections:
            graph.add_to_collection(key, self)
        self._save_slice_info = None
        
def restore_graph(graph_def, save_path=None,
                  saver_def=None,
                  input_map=None, return_elements=None, op_dict=None,
                  trainable=True, collections=None,
                 ):
    """
    Restore a graph from a GraphDef
    
    Args:
      graph_def: a GraphDef instance, representing the model architecture
      save_path: path where parameter values were saved
      saver_def: SaverDef for restoring the saver
      input_map, return_elements, op_dict: passed to tf.import_graph_def
      trainable: whether the restored variables should be marked as trainable
      collections: which collections to add the restored variables to
      
    Returns: (graph_elements, saver)
      graph_elements: The return value of tf.import_graph_def
      saver: The saver can be used to load further checkpoints
    """
    res = tf.import_graph_def(graph_def, name='', input_map=input_map, return_elements=return_elements, op_dict=op_dict)
    restored_vars = []
    for node in graph_def.node:
        if node.op == 'Variable':
            restored_vars.append(RestoredVariable(node.name, trainable=trainable, collections=collections))
    
    if saver_def is not None:
        saver = tf.train.Saver(saver_def, var_list=restored_vars)
    else:
        # Saver names must be unique, but we can't reuse the old saver variables without the saver_def
        # So we generate a random name, and hope the variable ordering and packing is deterministic and
        # unchanged since the checkpoint was saved
        saver = tf.train.Saver(var_list=restored_vars,
                               name='restored-' + ('%016x' % random.randrange(16**16)))
    
    if save_path is not None:
        saver.restore(tf.get_default_session(), save_path)
    
    return res, saver