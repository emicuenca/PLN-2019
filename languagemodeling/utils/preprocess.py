import os
import re

def main():
    filenames = ['test', 'corpus']
    regex = "<doc|<\/doc|$|ENDOFARTICLE|\s|-+|=+|Enlaces externos|Véase también|Referencias"
    prog = re.compile(regex)
    
    for filename in filenames:
        inFilename = filename + '.txt'
        outFilename = filename + '-utf8.txt'
        inFilepath = os.path.join(os.getcwd(), 'corpora', inFilename)
        outFilepath = os.path.join(os.getcwd(), 'corpora', outFilename)
        print(inFilepath, ">>", outFilepath)
        with open(inFilepath, 'r', encoding='cp1252') as f_in:
            lines = f_in.readlines()
            with open(outFilepath, 'w', encoding='utf-8') as f_out: 
                for line in lines:
                    if prog.match(line) is None:
                        f_out.write(line)
                    
if __name__ == '__main__':
    main()