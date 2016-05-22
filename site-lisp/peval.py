# -*- coding: utf-8 -*-
#
# Simple evaluation of one line of python code. Only eval() is called
# on the current line, and the result is displayed 2 line below,
# nothing else.
from Pymacs import lisp
import re, random
from turkish.deasciifier import Deasciifier

interactions = {}
glob = {}

def get_block_content(start_tag, end_tag):
    remember_where = lisp.point()
    block_begin = lisp.search_backward(start_tag)
    block_end = lisp.search_forward(end_tag)
    block_end = lisp.search_forward(end_tag)
    content = lisp.buffer_substring(block_begin, block_end)
    lisp.goto_char(remember_where)
    return block_begin, block_end, content

def pexec():
    global glob
    remember_where = lisp.point()
    block_begin, block_end, content = get_block_content("\n","\n")
    if "=" in content or "import" in content:
        c = compile(source=content,filename="",mode="single")
        eval(c,glob)
        print 'eval done'
    else:
        c = compile(source=content,filename="",mode="eval")
        res = eval(c,glob)
        lisp.forward_line(2)
        bb1, be2, bc2 = get_block_content("\n","\n")
        lisp.delete_region(bb1,be2)
        lisp.insert("\n")
        lisp.insert(str(res))
    lisp.goto_char(remember_where)
            
interactions[pexec] = ''

