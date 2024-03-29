#!/usr/bin/env python

from assignment.hierarchy import generate_hierarchy, test_hierarchy, generate_hierarchy_vis
from assignment.graph import get_parser
from nbdt.utils import maybe_install_wordnet


def main():
    maybe_install_wordnet()

    parser = get_parser()
    args = parser.parse_args()

    generate_hierarchy(**vars(args))
    test_hierarchy(args)
    generate_hierarchy_vis(args)

#python3 mnist_hierarchy.py --induced-linkage average --induced-affinity l1

if __name__ == '__main__':
    main()
