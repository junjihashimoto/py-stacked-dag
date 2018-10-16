#!/usr/bin/python
import sys
import stackeddag.graphviz as sd

def main ():
    argv = sys.argv
    argc = len(argv)
    if (argc != 2):
        print(f'Usage: python {argv[0]} arg1')
        quit()
    dot_file = argv[1]
    print(sd.fromDotFile(dot_file), end="")

if __name__ == '__main__':
    main()
