from typing import List, Tuple, Union, Optional

Vector = List[Union[int, float]]
Item = Tuple[Vector, int]  # (vector, original_index)


class KDNode:
    def __init__(self, item: Item, left=None, right=None):
        self.item = item
        self.left = left
        self.right = right


class KDTree:
    """
    KD-tree που χτίζεται πάνω σε items = [(vector, idx), ...]
    και επιστρέφει indices από range queries.
    """

    def __init__(self, items: List[Item], dimensions: List[str]):
        self.dimensions = dimensions
        self.k = len(dimensions)
        self.root = self._build_tree(items, depth=0)

    def _build_tree(self, items: List[Item], depth: int) -> Optional[KDNode]:
        if not items:
            return None

        axis = depth % self.k
        items.sort(key=lambda it: it[0][axis])  # sort by vector[axis]

        median_idx = len(items) // 2
        median_item = items[median_idx]

        left_items = items[:median_idx]
        right_items = items[median_idx + 1 :]

        return KDNode(
            item=median_item,
            left=self._build_tree(left_items, depth + 1),
            right=self._build_tree(right_items, depth + 1),
        )

    def query_range(self, lower_bounds: List[float], upper_bounds: List[float]) -> List[int]:
        bounds = [(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))]
        results: List[int] = []
        self._range_query(self.root, bounds, depth=0, results=results)
        return results

    def _range_query(self, node: Optional[KDNode], bounds: List[tuple], depth: int, results: List[int]):
        if node is None:
            return

        axis = depth % self.k
        vec, idx = node.item

        # check if inside bounds
        if all(bounds[i][0] <= vec[i] <= bounds[i][1] for i in range(self.k)):
            results.append(idx)

        # prune branches
        if node.left is not None and bounds[axis][0] <= vec[axis]:
            self._range_query(node.left, bounds, depth + 1, results)

        if node.right is not None and bounds[axis][1] >= vec[axis]:
            self._range_query(node.right, bounds, depth + 1, results)
