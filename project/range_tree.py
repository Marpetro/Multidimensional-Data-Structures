class RangeTreeNode:
    def __init__(self, points_sorted_x, points_sorted_y):
        self.points_sorted_x = points_sorted_x  # [(x,y,index)]
        self.points_sorted_y = points_sorted_y  # sorted by y
        self.left = None
        self.right = None
        self.mid_x = None


class RangeTree:
    def __init__(self, points_xy, indices):
        # points_xy = [(x,y), ...]
        # indices = [i1, i2, ...]

        pts = [(points_xy[i][0], points_xy[i][1], indices[i]) for i in range(len(points_xy))]
        pts_sorted_x = sorted(pts, key=lambda p: p[0])
        pts_sorted_y = sorted(pts, key=lambda p: p[1])

        self.root = self.build(pts_sorted_x, pts_sorted_y)

    def build(self, pts_x, pts_y):
        if not pts_x:
            return None

        node = RangeTreeNode(pts_x, pts_y)

        if len(pts_x) == 1:
            node.mid_x = pts_x[0][0]
            return node

        mid = len(pts_x) // 2
        node.mid_x = pts_x[mid][0]

        left_x = pts_x[:mid]
        right_x = pts_x[mid:]

        left_set = set(left_x)

        left_y = [p for p in pts_y if p in left_set]
        right_y = [p for p in pts_y if p not in left_set]

        node.left = self.build(left_x, left_y)
        node.right = self.build(right_x, right_y)

        return node

    def range_query(self, node, x_low, x_high, y_low, y_high, results):
        if node is None:
            return

        # If full subtree inside x-range → use y-list
        if x_low <= node.points_sorted_x[0][0] and node.points_sorted_x[-1][0] <= x_high:
            # Do binary search on y-sorted list
            for (x, y, idx) in node.points_sorted_y:
                if y_low <= y <= y_high:
                    results.append(idx)
            return

        # Otherwise, check children
        mid = node.mid_x

        if x_low <= mid:
            self.range_query(node.left, x_low, x_high, y_low, y_high, results)

        if x_high >= mid:
            self.range_query(node.right, x_low, x_high, y_low, y_high, results)
