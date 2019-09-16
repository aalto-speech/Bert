#!/usr/bin/env python3

import logging
import sys
import os
import lzma
import random

def split(inp_f, target_lines, l, of1, of2):
    of1 = open(of1, 'w', encoding='utf-8')
    of2 = open(of2, 'w', encoding='utf-8')
    print("{} lines".format(l))
    p = target_lines / l
    p *= 1.05
    if p > 0.1:
        p = 0.1
    print("using {} as percentage".format(p))

    c = 0
    for line in open(inp_f, encoding='utf-8'):
        line = line.strip()
        if not line.startswith("<s>"):
            line = "<s> " + line
        if not line.endswith("</s>"):
            line = line + " </s>"

        if random.random() < p and c < target_lines:
            print(line, file=of1)
            c += 1
        else:
            print(line, file=of2)



def parse_name(d):
    base = os.path.basename(d)
    assert base.startswith("varik")
    parts = base.split('_')b 
    btype = parts[1]
    #order = int(parts[2][1:])
    return btype

def main(d):
    btype = parse_name(d)
#    open(os.path.join(d, 'order'), 'w', encoding='utf-8').write("{}\n".format(order)) 
    
    word_map = os.path.join(os.path.dirname(d),'wordmap')

    between = " "
    prefix = ""
    suffix = ""
   
    assert btype in {"aff", "wma", "suf", "pre", "word"}
    if btype == "wma":
        between = " <w> "
    if btype == "pre" or btype == "aff":
        prefix ="+"
    if btype == "suf" or btype == "aff":
        suffix ="+"

    m = {}
    for line in open(word_map, encoding='utf-8'):
        parts = line.strip().split()
        m[parts[0]] = "{} {}".format(suffix,prefix).join(parts[1:]).replace("+<unk>","<unk>").replace("<unk>+", "<unk>").replace("+<UNK>","<UNK>").replace("<UNK>+", "<UNK>").replace("<unk>", "<UNK>")
        if "<UNK>" in m[parts[0]]:
            m[parts[0]] = "<UNK>"
    m["<s>"] = "<s>"
    m["</s>"] = "</s>"
    ddir = os.path.dirname(os.path.dirname(d)) 
    for f in os.listdir(ddir):
        lc = 0
        if not f.endswith(".xz"):
            continue
        if os.path.exists(os.path.join(d, f[:-3])):
            continue
        with open(os.path.join(d, f[:-3]), 'w', encoding='utf-8') as outf:
            for line in lzma.open(os.path.join(ddir, f), 'rt', encoding='utf-8'):
                parts = line.strip().split()
                for p in parts:
                    if p not in m:
                        m[p] = "<UNK>"
                new_line = between + between.join(m[p] for p in parts) + between
                new_line = new_line.strip()
                if not new_line.startswith("<s>"):
                    new_line = "<s> " + new_line
                if not new_line.endswith("</s>"):
                    new_line = new_line + " </s>"
                while new_line.startswith("<s> <w> <s> <w> "):
                    new_line = new_line[len("<s> <w> "):]
                while new_line.endswith("</s> <w> </s>"):
                    new_line = new_line[:-len(" <w> </s>"):]
                print(new_line.strip(), file=outf)
                lc += 1
        split(os.path.join(d, f[:-3]), 10000, lc, os.path.join(d, f[:-3] + ".dev"), os.path.join(d, f[:-3] + ".train"))

            

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1])
