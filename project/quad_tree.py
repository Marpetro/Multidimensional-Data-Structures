class QuadTreeNode:
    def __init__(self, x_min, x_max, y_min, y_max, capacity=10):
        self.bounds = (x_min, x_max, y_min, y_max)
        self.capacity = capacity
        self.points = []      # will hold tuples (x, y, index)
        self.children = None  # subdivided nodes

    def subdivide(self):
        x_min, x_max, y_min, y_max = self.bounds
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2

        self.children = [
            QuadTreeNode(x_min, x_mid, y_min, y_mid),  # bottom-left
            QuadTreeNode(x_mid, x_max, y_min, y_mid),  # bottom-right
            QuadTreeNode(x_min, x_mid, y_mid, y_max),  # top-left
            QuadTreeNode(x_mid, x_max, y_mid, y_max),  # top-right
        ]


class QuadTree:
    def __init__(self, points_xy, indices, capacity=10):
        # points_xy = [(x,y), ...]
        # indices   = [original_index]
        xs = [p[0] for p in points_xy]
        ys = [p[1] for p in points_xy]

        self.root = QuadTreeNode(
            min(xs), max(xs),
            min(ys), max(ys),
            capacity
        )

        for p, idx in zip(points_xy, indices):
            self.insert(self.root, p[0], p[1], idx)

    def insert(self, node, x, y, index):
        x_min, x_max, y_min, y_max = node.bounds

        # Ignore if outside node bounds
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            return

        # If node has space → insert
        if len(node.points) < node.capacity and node.children is None:
            node.points.append((x, y, index))
            return

        # Otherwise subdivide if needed
        if node.children is None:
            node.subdivide()

        # Insert into the correct child
        for child in node.children:
            self.insert(child, x, y, index)

    def range_query(self, node, x_low, x_high, y_low, y_high, results):
        if node is None:
            return

        x_min, x_max, y_min, y_max = node.bounds

        # If there's no overlap
        if x_high < x_min or x_low > x_max or y_high < y_min or y_low > y_max:
            return

        # Check points in node
        for (x, y, idx) in node.points:
            if x_low <= x <= x_high and y_low <= y <= y_high:
                results.append(idx)

        # Recurse into children
        if node.children:
            for child in node.children:
                self.range_query(child, x_low, x_high, y_low, y_high, results)
