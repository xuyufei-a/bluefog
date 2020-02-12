from typing import List, Dict
import torch

from bluefog.torch import mpi_lib  # C library
from bluefog.common.basics import BlueFogBasics

_basics = BlueFogBasics(__file__, 'mpi_lib')

# import basic methods
init = _basics.init
shutdown = _basics.shutdown
size = _basics.size
local_size = _basics.local_size
rank = _basics.rank
local_rank = _basics.local_rank
mpi_threads_supported = _basics.mpi_threads_supported
load_topology = _basics.load_topology
set_topology = _basics.set_topology
in_neighbor_ranks = _basics.in_neighbor_ranks
out_neighbor_ranks = _basics.out_neighbor_ranks

# Schema: handle -> input, output
# We keep input in order to make sure it does not get garbage collected
# before the operation is finished.
_handle_map = {}

# Schema: name -> tensor
_win_map = {}

#Schema: handle -> name
_win_handle_map = {}


def _check_function(function_factory, tensor, *args):
    function = function_factory(tensor, *args)
    if not hasattr(mpi_lib, function):
        raise ValueError('Tensor type %s is not supported.' % tensor.type())
    if not tensor.is_contiguous():
        raise ValueError('Tensor is required to be contiguous.')
    return function


def _allreduce_function_factory(tensor):
    return 'bluefog_torch_allreduce_async_' + tensor.type().replace('.', '_')


def _allreduce_async(tensor, output, average, name):
    function = _check_function(_allreduce_function_factory, tensor)
    if average:
        assert isinstance(tensor, (torch.FloatTensor, torch.DoubleTensor,
                                   torch.cuda.FloatTensor, torch.cuda.DoubleTensor)), \
            "If average is set in allreduce, only float or double tensor is allowed."

    handle = getattr(mpi_lib, function)(tensor, output, average,
                                        name.encode() if name is not None else "")
    _handle_map[handle] = (tensor, output)
    return handle


def allreduce(tensor: torch.Tensor, average: bool = True, name: str = None) -> torch.Tensor:
    """
    A function that performs averaging or summation of the input tensor over all the
    Bluefog processes. The input tensor is not modified.

    The reduction operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    Bluefog processes for a given name. The reduction will not start until all processes
    are ready to send and receive the tensor.

    Arguments:
        tensor: A tensor to average and sum.
        average: A flag indicating whether to compute average or summation,
                 defaults to average.
        name: A name of the reduction operation.

    Returns:
        A tensor of the same shape and type as `tensor`, averaged or summed across all
        processes.
    """
    handle = allreduce_async(tensor, average, name)
    return synchronize(handle)


def allreduce_async(tensor: torch.Tensor, average: bool = True, name: str = None) -> int:
    """
    A function that performs asynchronous averaging or summation of the input tensor
    over all the Bluefog processes. The input tensor is not modified.

    The reduction operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    Bluefog processes for a given name. The reduction will not start until all processes
    are ready to send and receive the tensor.

    Arguments:
        tensor: A tensor to average and sum.
        average: A flag indicating whether to compute average or summation,
                 defaults to average.
        name: A name of the reduction operation.

    Returns:
        A handle to the allreduce operation that can be used with `poll()` or
        `synchronize()`.
    """
    output = tensor.new(tensor.shape)
    return _allreduce_async(tensor, output, average, name)


def _broadcast_function_factory(tensor):
    return 'bluefog_torch_broadcast_async_' + tensor.type().replace('.', '_')


def _broadcast_async(tensor, output, root_rank, name):
    function = _check_function(_broadcast_function_factory, tensor)
    handle = getattr(mpi_lib, function)(tensor, output, root_rank,
                                        name.encode() if name is not None else "")
    _handle_map[handle] = (tensor, output)
    return handle


def broadcast(tensor: torch.Tensor, root_rank: int, name: str = None) -> torch.Tensor:
    """
    A function that broadcasts the input tensor on root rank to the same input tensor
    on all other Bluefog processes. The input tensor is not modified.

    The broadcast operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    Bluefog processes for a given name. The broadcast will not start until all processes
    are ready to send and receive the tensor.

    This acts as a thin wrapper around an autograd function.  If your input
    tensor requires gradients, then callings this function will allow gradients
    to be computed and backpropagated.

    Arguments:
        tensor: A tensor to broadcast.
        root_rank: The rank to broadcast the value from.
        name: A name of the broadcast operation.

    Returns:
        A tensor of the same shape and type as `tensor`, with the value broadcasted
        from root rank.
    """
    handle = broadcast_async(tensor, root_rank, name)
    return synchronize(handle)


def broadcast_async(tensor: torch.Tensor, root_rank: int, name: str = None) -> int:
    """
    A function that asynchronously broadcasts the input tensor on root rank to the same
    input tensor on all other Bluefog processes. The input tensor is not modified.

    The broadcast operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    Bluefog processes for a given name. The broadcast will not start until all processes
    are ready to send and receive the tensor.

    Arguments:
        tensor: A tensor to broadcast.
        root_rank: The rank to broadcast the value from.
        name: A name of the broadcast operation.

    Returns:
        A handle to the broadcast operation that can be used with `poll()` or
        `synchronize()`.
    """
    output = tensor.new(tensor.shape)
    return _broadcast_async(tensor, output, root_rank, name)


def broadcast_(tensor, root_rank, name=None) -> torch.Tensor:
    """
    A function that broadcasts the input tensor on root rank to the same input tensor
    on all other Bluefog processes. The operation is performed in-place.

    The broadcast operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    Bluefog processes for a given name. The broadcast will not start until all processes
    are ready to send and receive the tensor.

    Arguments:
        tensor: A tensor to broadcast.
        root_rank: The rank to broadcast the value from.
        name: A name of the broadcast operation.

    Returns:
        A tensor of the same shape and type as `tensor`, with the value broadcasted
        from root rank.
    """
    handle = broadcast_async_(tensor, root_rank, name)
    return synchronize(handle)


def broadcast_async_(tensor, root_rank, name=None) -> int:
    """
    A function that asynchronously broadcasts the input tensor on root rank to the same
    input tensor on all other Bluefog processes. The operation is performed in-place.

    The broadcast operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    Bluefog processes for a given name. The broadcast will not start until all processes
    are ready to send and receive the tensor.

    Arguments:
        tensor: A tensor to broadcast.
        root_rank: The rank to broadcast the value from.
        name: A name of the broadcast operation.

    Returns:
        A handle to the broadcast operation that can be used with `poll()` or
        `synchronize()`.
    """
    return _broadcast_async(tensor, tensor, root_rank, name)


def _allgather_function_factory(tensor):
    return 'bluefog_torch_allgather_async_' + tensor.type().replace('.', '_')


def _allgather_async(tensor, output, name):
    function = _check_function(_allgather_function_factory, tensor)
    handle = getattr(mpi_lib, function)(tensor, output,
                                        name.encode() if name is not None else "")
    _handle_map[handle] = (tensor, output)
    return handle


def allgather(tensor: torch.Tensor, name: str = None) -> torch.Tensor:
    """
    A function that concatenates the input tensor with the same input tensor on
    all other Bluefog processes. The input tensor is not modified.

    The concatenation is done on the first dimension, so the input tensors on the
    different processes must have the same rank and shape.

    Arguments:
        tensor: A tensor to allgather.
        name: A name of the allgather operation.

    Returns:
        A tensor of the same type as `tensor`, concatenated on dimension zero
        across all processes. The shape is identical to the input shape, except for
        the first dimension, which may be greater and is the sum of all first
        dimensions of the tensors in different Bluefog processes.
    """
    handle = allgather_async(tensor, name)
    return synchronize(handle)


def allgather_async(tensor: torch.Tensor, name: str = None) -> int:
    """
    A function that asynchronously concatenates the input tensor with the same input
    tensor on all other Bluefog processes. The input tensor is not modified.

    The concatenation is done on the first dimension, so the input tensors on the
    different processes must have the same rank and shape.

    Arguments:
        tensor: A tensor to allgather.
        name: A name of the allgather operation.

    Returns:
        A handle to the allgather operation that can be used with `poll()` or
        `synchronize()`.
    """
    output = tensor.new()  # real size will be allocated later.
    return _allgather_async(tensor, output, name)


def _neighbor_allgather_function_factory(tensor):
    return 'bluefog_torch_neighbor_allgather_async_' + tensor.type().replace('.', '_')


def _neighbor_allgather_async(tensor, output, name):
    function = _check_function(_neighbor_allgather_function_factory, tensor)
    handle = getattr(mpi_lib, function)(tensor, output,
                                        name.encode() if name is not None else "")
    _handle_map[handle] = (tensor, output)
    return handle


def neighbor_allgather(tensor: torch.Tensor, name: str = None) -> torch.Tensor:
    """
    A function that concatenates the input tensor with the same input tensor on
    on all neighbor Bluefog processes (Not include self). The input tensor is not modified.

    The concatenation is done on the first dimension, so the input tensors on the
    different processes must have the same rank and shape.

    Arguments:
        tensor: A tensor to allgather.
        name: A name of the allgather operation.

    Returns:
        A tensor of the same type as `tensor`, concatenated on dimension zero
        across all processes. The shape is identical to the input shape, except for
        the first dimension, which may be greater and is the sum of all first
        dimensions of the tensors in neighbor Bluefog processes.
    """
    handle = neighbor_allgather_async(tensor, name)
    return synchronize(handle)


def neighbor_allgather_async(tensor: torch.Tensor, name: str = None) -> int:
    """
    A function that asynchronously concatenates the input tensor with the same input
    tensor on all neighbor Bluefog processes (Not include self).
    The input tensor is not modified.

    The concatenation is done on the first dimension, so the input tensors on the
    different processes must have the same rank and shape.

    Arguments:
        tensor: A tensor to allgather.
        name: A name of the allgather operation.

    Returns:
        A handle to the allgather operation that can be used with `poll()` or
        `synchronize()`.
    """
    output = tensor.new()  # real size will be allocated later.
    return _neighbor_allgather_async(tensor, output, name)


def _neighbor_allreduce_function_factory(tensor):
    return 'bluefog_torch_neighbor_allreduce_async_' + tensor.type().replace('.', '_')


def _neighbor_allreduce_async(tensor, output, average, name):
    function = _check_function(_neighbor_allreduce_function_factory, tensor)
    if average:
        assert isinstance(tensor, (torch.FloatTensor, torch.DoubleTensor,
                                   torch.cuda.FloatTensor, torch.cuda.DoubleTensor)), \
            "If average is set in allreduce, only float or double tensor is allowed."
    handle = getattr(mpi_lib, function)(tensor, output, average,
                                        name.encode() if name is not None else "")
    _handle_map[handle] = (tensor, output)
    return handle


def neighbor_allreduce(tensor: torch.Tensor, average: bool = True,
                       name: str = None) -> torch.Tensor:
    """
    A function that performs averaging or summation of the input tensor
    over the negihbors in the Bluefog processes, where neighbors always include the itself.
    The input tensor is not modified.

    The reduction operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    Bluefog processes for a given name. The reduction will not start until all processes
    are ready to send and receive the tensor.

    Arguments:
        tensor: A tensor to average and sum.
        average: A flag indicating whether to compute average or summation,
                 defaults to average.
        name: A name of the reduction operation.

    Returns:
        A tensor of the same shape and type as `tensor`, averaged or summed across all
        processes.
    """
    handle = neighbor_allreduce_async(tensor, average, name)
    return synchronize(handle)


def neighbor_allreduce_async(tensor: torch.Tensor, average: bool = True, name: str = None) -> int:
    """
    A function that asynchronously averaging or summation of the input tensor
    over the negihbors in the Bluefog processes, where neighbors always include the itself.
    The input tensor is not modified.

    The reduction operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    Bluefog processes for a given name. The reduction will not start until all processes
    are ready to send and receive the tensor.

    Arguments:
        tensor: A tensor to neighbor_allreduce.
        name: A name of the neighbor_allreduce operation.

    Returns:
        A handle to the neighbor_allreduce operation that can be used with `poll()` or
        `synchronize()`.
    """
    output = tensor.new(tensor.shape)
    return _neighbor_allreduce_async(tensor, output, average, name)


def poll(handle: int) -> bool:
    """
    Polls an allreduce, allgather or broadcast handle to determine whether underlying
    asynchronous operation has completed. After `poll()` returns `True`, `synchronize()`
    will return without blocking.

    Arguments:
        handle: A handle returned by an allreduce, allgather or broadcast asynchronous
                operation.

    Returns:
        A flag indicating whether the operation has completed.
    """
    return mpi_lib.bluefog_torch_poll(handle) != 0


def synchronize(handle: int) -> torch.Tensor:
    if handle not in _handle_map:
        return None
    mpi_lib.bluefog_torch_wait_and_clear(handle)
    _, output = _handle_map.pop(handle)
    return output


def barrier():
    """ Barrier function to stop and sychronize all MPI processes."""
    return mpi_lib.bluefog_torch_barrier();

# MPI one sided ops, which will be useful in the asynchronized algorithm.
def _win_create_function_factory(tensor):
    return 'bluefog_torch_win_create_' + tensor.type().replace('.', '_')


def win_create(tensor: torch.Tensor, name: str) -> bool:
    """ Create MPI window for remote memoery access. The window is dedicated to
    the provided tensor only, which is identified by unqiue name. It is blocking operations.
    The initial value of MPI windows for neighbor is the same as input tensor.

    Args:
        tensor (torch.Tensor): Provide the size, data type, and/or memory for window.
        name (str): The unique name to associate the window object.

    Returns:
        bool: Indicate the creation succeed or not.
    """
    # TODO(ybc): How to make sure that different ranks
    # create window wtih same name and size?
    function = _check_function(_win_create_function_factory, tensor)
    if getattr(mpi_lib, function)(tensor, name):
        _win_map[name] = tensor
        return True
    return False


def win_free(name: str = None) -> bool:
    """ Free the MPI windows associated with name.

    Args:
        name (str): The unique name to associate the window object.
            If name is none, free all the window objects.

    Returns:
        bool: Indicate the free succeed or not.
    """
    if name is None:
        _win_map.clear()
        name = ''
    else:
        _win_map.pop(name)
    return getattr(mpi_lib, 'bluefog_torch_win_free')(name)


def _win_sync_function_factory(tensor, weights):
    return ('bluefog_torch_win_sync_'
            + ('with_weights_' if weights else '')
            + tensor.type().replace('.', '_'))


def win_sync(name: str, weights: Dict[int, float] = None) -> torch.Tensor:
    """Locally synchronized the window objects and returned the reduced neighbor tensor.

    Args:
        name: The unique name to associate the window object.
        weights: If weights is presented, the return tensor will return the weighted average
            defined by this weights. The data structure of weights should be {rank : weight}
            and rank has to belonge the (in-)neighbors and self.

    Returns:
        torch.Tensor: The average tensor of all neighbors' cooresponding tensors.

    Note: Weights here will be useful if you need a dynamic weighted average, i.e. the weights
    change with the iterations. If static weight need, then setting the weights through the
    win_create is a better choice. TODO(ybc) add weights setting in win_create.
    """
    tensor = _win_map[name]
    function = _check_function(_win_sync_function_factory, tensor, weights)
    if weights is not None:
        # Pre-condition check for weights dictionary.
        if not isinstance(weights, dict):
            raise ValueError("Argument weights has to be a dictionary map from the (in-)neighbor "
                             "rank to the weights.")
        if not set(weights.keys()).issubset(set(in_neighbor_ranks() + [rank()])):
            raise ValueError("The key of weights should only contain the ranks that belong to "
                             " in-neighbors and self ranks.")

        if not getattr(mpi_lib, function)(tensor, name, weights):
            raise RuntimeError("Cannot apply win_sync on " + name)
        return tensor

    if not getattr(mpi_lib, function)(tensor, name):
        raise RuntimeError("Cannot apply win_sync on " + name)
    return tensor


def win_fence(name: str) -> bool:
    """ A collective call to synchronization on MPI window with associated name.

    Warning: The API win_get and win_put provied here is already wrapped by
    MPI_Win_lock and MPI_Win_unlock. So you should not explicitly call win_fence there.
    """
    return mpi_lib.bluefog_torch_win_fence(name)


def _win_put_function_factory(tensor):
    return 'bluefog_torch_win_put_' + tensor.type().replace('.', '_')


def win_put(tensor: torch.Tensor, name: str,
            dst_ranks: List[int] = None) -> int:
    """ Passively put the tensor into neighbor's shared window memory.
    This is a non-blocking function, which will return without waiting the
    win_put operation is really finished.

    Args:
        tesnor: The tensor that shares to neighbor.
        name: The unique name to associate the window object.
        dst_ranks: The source ranks to put the value for. If not provided, it will
            put into all neighbors' shared memory defined by virtual topology.
            dst_ranks should only contain the ranks that belong to out-neighbors.

    Returns:
        A handle to the allgather operation that can be used with `win_poll()` or
        `win_wait()`.
    """
    function = _check_function(_win_put_function_factory, tensor)
    dst_ranks = out_neighbor_ranks() if dst_ranks is None else dst_ranks
    if not set(dst_ranks).issubset(set(out_neighbor_ranks())):
        raise ValueError(
            "dst_ranks should only contain the ranks that belong to out-neighbors.")
    handle = getattr(mpi_lib, function)(tensor, name, dst_ranks)
    _win_handle_map[handle] = name
    return handle


def win_put_blocking(tensor: torch.Tensor, name: str,
                     dst_ranks: List[int] = None) -> bool:
    """ Passively put the tensor into neighbor's shared window memory.
    This is a blocking function, which will return until win_put operation
    is finished.

    Args:
        tensor: The tensor that shares to neighbor.
        name: The unique name to associate the window object.
        dst_ranks: The source ranks to get the value from. If not provided, it will
            put into all neighbors' shared memory defined by virtual topology.
            dst_ranks should only contain the ranks that belong to out-neighbors.

    Returns:
        A bool value to indicate the put succeeded or not.
    """
    handle = win_put(tensor, name, dst_ranks)
    win_wait(handle)
    # TODO(ybc) Error handling.
    return True


def _win_get_function_factory(tensor):
    return 'bluefog_torch_win_get_' + tensor.type().replace('.', '_')


def win_get(tensor: torch.Tensor, name: str,
            src_ranks: List[int] = None, average: bool = True) -> int:
    """ Passively get the tensor(s) from neighbors' shared window memory into
    local memory. The input tensor is also in-place output.

    Args:
        tensor: A tensor to get the result, should have same shape and type of
            the window object associated with name.
        name: The unique name to associate the window object.
        src_ranks: The source ranks to get the value from. If not provided, it will
            get all neighbors' values defined by virtual topology.
            src_ranks should only contain the ranks that belong to in-neighbors.
        average: A flag indicating whether to compute average or summation,
                 defaults to average.

    Returns:
        A handle to the allgather operation that can be used with `poll()` or
        `synchronize()`.
    """
    function = _check_function(_win_get_function_factory, tensor)
    src_ranks = in_neighbor_ranks() if src_ranks is None else src_ranks
    if not set(src_ranks).issubset(set(in_neighbor_ranks())):
        raise ValueError(
            "src_ranks should only contain the ranks that belong to in-neighbors.")
    handle = getattr(mpi_lib, function)(
        tensor, name, src_ranks, average)
    _win_handle_map[handle] = name
    return handle


def win_get_blocking(tensor: torch.Tensor, name: str,
                     src_ranks: List[int] = None, average: bool = True) -> bool:
    """ Passively get the tensor(s) from neighbors' shared window memory into
    local memory. The input tensor is also in-place output.

    Args:
        tensor: A tensor to get the result, should have same shape and type of
            the window object associated with name.
        name: The unique name to associate the window object.
        src_ranks: The source ranks to get the value from. If not provided, it will
            get all neighbors' values defined by virtual topology.
            src_ranks should only contain the ranks that belong to in-neighbors.
        average: A flag indicating whether to compute average or summation,
                 defaults to average.

    Returns:
        A tensor of the same shape and type as `tensor`, averaged or summed across src_ranks
        processes (or all neighbor processes).
    """
    handle = win_get(tensor, name, src_ranks, average)
    win_wait(handle)
    # TODO(ybc) Error handling.
    return True


def win_poll(handle: int) -> bool:
    return mpi_lib.bluefog_torch_win_poll(handle) != 0


def win_wait(handle: int) -> bool:
    if handle not in _handle_map:
        return None
    mpi_lib.bluefog_torch_win_wait(handle)
    _ = _win_handle_map.pop(handle)
    return True
