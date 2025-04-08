import itertools

import jax.numpy as jnp
import numpy as np
from pykronecker import KroneckerProduct

from kreg.kernel.component import KernelComponent
from kreg.kernel.dimension import Dimension
from kreg.typing import DataFrame, JAXArray, List
from kreg.utils import memory_profiled, logger


class KroneckerKernel:
    """Kronecker product of all kernel functions to form a complete kernel
    linear mapping.

    Parameters
    ----------
    kernels
        List of kernel functions.
    grids
        List of value grids, unique values for each dimension.
    nugget
        Regularization for the kernel matrix.

    """

    def __init__(
        self,
        kernel_components: "List[KernelComponent]",
        nugget: float = 5e-8,
    ) -> None:
        self.kernel_components = kernel_components
        self.nugget = nugget

        self.dimensions: list[Dimension] = []
        self.columns: list[str] = []
        for component in self.kernel_components:
            dimensions = component.dimensions
            if isinstance(dimensions, Dimension):
                self.dimensions.append(dimensions)
            else:
                self.dimensions.extend(dimensions)
        for dimension in self.dimensions:
            columns = dimension.columns
            if isinstance(columns, str):
                self.columns.append(columns)
            else:
                self.columns.extend(columns)

        self.kmats: list[JAXArray]
        self.op_k: KroneckerProduct
        self.eigdecomps: list[tuple[JAXArray, JAXArray]]
        self.op_p: KroneckerProduct
        self.op_root_k: KroneckerProduct
        self.op_root_p: KroneckerProduct
        self.status = "detached"

    @property
    def span(self) -> DataFrame:
        columns = list(
            itertools.chain.from_iterable(
                [dim.columns for dim in self.dimensions]
            )
        )
        span = DataFrame(
            data=np.array(
                [
                    np.hstack(list(row))
                    for row in itertools.product(
                        *[dim.span for dim in self.dimensions]
                    )
                ]
            ),
            columns=columns,
        )
        return span

    @memory_profiled
    def _build_matrices(self):
        """Build the kernel matrices and their eigendecompositions.
        This is a memory-intensive operation for large dimension spans."""
        logger.debug("Building kernel matrices")
        
        self.kmats = [
            component.build_kmat(self.nugget)
            for component in self.kernel_components
        ]
        
        # Log matrix sizes to assist with memory profiling
        for i, kmat in enumerate(self.kmats):
            logger.debug(f"Component {i} matrix size: {kmat.shape}, dtype: {kmat.dtype}")
        
        logger.debug("Building KroneckerProduct")
        self.op_k = KroneckerProduct(self.kmats)
        
        logger.debug("Computing eigendecompositions")
        self.eigdecomps = list(map(jnp.linalg.eigh, self.kmats))
        
        logger.debug("Building precision matrix")
        self.op_p = KroneckerProduct(
            [(mat / vec).dot(mat.T) for vec, mat in self.eigdecomps]
        )
        
        logger.debug("Building root matrices")
        self.op_root_k = KroneckerProduct(
            [(mat * jnp.sqrt(vec)).dot(mat.T) for vec, mat in self.eigdecomps]
        )
        self.op_root_p = KroneckerProduct(
            [(mat / jnp.sqrt(vec)).dot(mat.T) for vec, mat in self.eigdecomps]
        )
        
        # Calculate approximate memory usage - rough estimation
        total_size = 0
        for kmat in self.kmats:
            # Each element is a float64 (8 bytes)
            total_size += kmat.size * 8
        
        # The eigendecompositions and derived matrices approximately
        # multiply this by a factor (very rough estimation)
        total_size_mb = total_size * 4 / (1024 * 1024)
        
        logger.info(f"Estimated matrix memory usage: {total_size_mb:.2f} MB")

    @memory_profiled
    def attach(self, data: DataFrame) -> None:
        """Attach data to the kernel and build matrices.
        
        Parameters
        ----------
        data : DataFrame
            The data to attach to the kernel
        """
        if self.status == "detached":
            logger.info(f"Attaching data with shape {data.shape}")
            
            # Log dimension sizes before setting spans
            for i, component in enumerate(self.kernel_components):
                dims = component.dimensions if isinstance(component.dimensions, list) else [component.dimensions]
                for j, dim in enumerate(dims):
                    logger.debug(f"Component {i}, dimension {j} name: {dim.name}, columns: {dim.columns}")
            
            # Set spans for each component
            for component in self.kernel_components:
                component.set_span(data)
            
            # Log dimension sizes after setting spans
            for i, dim in enumerate(self.dimensions):
                if hasattr(dim, 'span') and hasattr(dim.span, 'shape'):
                    logger.debug(f"Dimension {i} ({dim.name}) span size: {len(dim.span)}")
            
            # Build matrices
            self._build_matrices()
            self.status = "attached"
            
            # Estimate full kernel size
            full_size = 1
            for dim in self.dimensions:
                if hasattr(dim, 'span'):
                    full_size *= len(dim.span)
            
            logger.info(f"Full kernel dimensions: {full_size} x {full_size} (virtual size)")

    @memory_profiled
    def clear_matrices(self) -> None:
        """Clear all matrices to free memory."""
        if self.status == "attached":
            logger.debug("Clearing kernel matrices to free memory")
            del self.kmats
            del self.op_k
            del self.eigdecomps
            del self.op_p
            del self.op_root_k
            del self.op_root_p
            self.status = "detached"

    @memory_profiled
    def dot(self, x: JAXArray) -> JAXArray:
        """Apply the kernel to a vector.
        
        Parameters
        ----------
        x : JAXArray
            Vector to multiply with the kernel
            
        Returns
        -------
        JAXArray
            Result of kernel multiplication
        """
        logger.debug(f"Applying kernel to vector of shape {x.shape}")
        return self.op_k @ x

    def __matmul__(self, x: JAXArray) -> JAXArray:
        return self.dot(x)

    def __len__(self) -> int:
        return len(self.span)
