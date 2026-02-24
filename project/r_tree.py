from rtree import index

class RTree:
    def __init__(self, vectors_5d, indices):
        # vectors_5d = [[d1,d2,d3,d4,d5], ...]
        p = index.Property()
        p.dimension = 5
        self.idx = index.Index(properties=p)

        for i, vec in enumerate(vectors_5d):
            idx_val = indices[i]
            d1, d2, d3, d4, d5 = vec
            # point as 5D rectangle: (min coords..., max coords...)
            self.idx.insert(idx_val, (d1, d2, d3, d4, d5, d1, d2, d3, d4, d5))

    def range_query_5d(self, lower_bounds, upper_bounds):
        # lower_bounds = [l1..l5], upper_bounds = [u1..u5]
        l1, l2, l3, l4, l5 = lower_bounds
        u1, u2, u3, u4, u5 = upper_bounds
        return list(self.idx.intersection((l1, l2, l3, l4, l5, u1, u2, u3, u4, u5)))
    
    
