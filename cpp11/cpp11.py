#!/usr/bin/env python3
#
# Copyright (c) 2011, Kenny Chan <kennytm@gmail.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import re
from itertools import zip_longest, tee, chain
from collections import namedtuple

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.addnodes import *
from sphinx.roles import XRefRole
from sphinx.locale import l_, _
from sphinx.domains import Domain, ObjType
from sphinx.directives import ObjectDescription
from sphinx.util.nodes import make_refnode
from sphinx.util.docfields import Field

class CPP11Error(Exception):
    pass

Token = namedtuple('Token', ['s', 't'])

TOK_IDENT = 'identifier'
TOK_KEYWORD = 'keyword'
TOK_SYMBOL = 'symbol'
TOK_LITERAL = 'literal'

REGEXES = (
    (None, re.compile(r'\s+')),
    (TOK_SYMBOL, re.compile(r'''
        \.{3} | ->\*? | >>= |
        [\#:]{1,2} |
        \.\*? |
        [-+*/%^&|!=<]=? |
        [{}\[\]();?~,<>]    # for simplicity, we regard '>>' as 2 tokens.
    ''', re.X)),
    (TOK_KEYWORD, re.compile(r'''\b(?:
        alignas | alignof | asm | auto | bool | break | case | catch | char16_t |
        char32_t | char | class | constexpr | const_cast | const | continue |
        decltype | default | delete | double | dynamic_cast | do | else | enum |
        explicit | export | extern | false | float | for | friend | goto | if |
        inline | int | long | mutable | namespace | new | noexcept | nullptr |
        operator | private | protected | public | register | reinterpret_cast |
        return | short | signed | sizeof | static_assert | static_cast | static |
        struct | switch | template | this | thread_local | throw | true | try |
        typedef | typeid | typename | union | unsigned | using | virtual | void |
        volatile | wchar_t | while |
        and_eq | and | bitand | bitor | compl | not_eq | not | or_eq | or |
        xor_eq | xor
    )\b''', re.X)),
    (TOK_LITERAL, re.compile(r'''
        (?: (?:\d*\.\d+|\d+\.) (?:[eE][-+]?\d+) | (?:\d+[eE][-+]?\d+) ) \w* |
        \d+\w* |
        (?:u8 | [uUL])? (?:
            " (?:[^"\n\\]|\\.)* " |
            ' (?:[^'\n\\]|\\.)* ' |
            R" ([^\s()\\]*) \( .*? \) \1 "
        )
    ''', re.X)),
    (TOK_IDENT, re.compile(r'\b[a-z_]\w*\b', re.I))
)

def tokenize(source: str) -> iter([Token]):
    index = 0
    length = len(source)
    while index < length:
        for toktype, regex in REGEXES:
            m = regex.match(source, index)
            if m:
                if toktype:
                    yield Token(m.group(), toktype)
                index = m.end()
                break
        else:
            raise CPP11Error("Cannot tokenize: " + source[index:])


NO_SPACE_AFTER = {
    '{', '[', '#', '(', '::', '.', '.*', '->*', '->', '<',
}
NO_SPACE_BEFORE = {
    '}', ']', ')', ';', '...', '::', '.', '.*', ',', '->*', '->', '>',
    '*', '&', '&&'
}
BRACKETS = {'(', '[', '{', '<', '>', '}', ']', ')'}

def untokenize(tokens: iter([Token]), callback=None) -> str:
    def add_spaces():
        a, b = tee(tokens)
        next(b, None)
        for i, ((s0, t0), (s1, t1)) in enumerate(zip_longest(a, b, fillvalue=(None, None))):
            yield (i, s0)
            if t0 == TOK_SYMBOL and s0 in NO_SPACE_AFTER:
                continue
            if t1 == TOK_SYMBOL and s1 in NO_SPACE_BEFORE:
                continue
            if t0 == TOK_SYMBOL and t1 == TOK_SYMBOL and s0 in BRACKETS and s1 in BRACKETS:
                continue
            if t0 != TOK_SYMBOL and t1 == TOK_SYMBOL and s1 in BRACKETS:
                continue
            if t1 != TOK_SYMBOL and t0 == TOK_SYMBOL and s0 in BRACKETS:
                continue
            if t0 == TOK_KEYWORD and s0 == 'operator' and t1 == TOK_SYMBOL:
                continue
            if t1 is None:
                continue
            yield (-1, ' ')

    if callback is None:
        callback = lambda x: ''.join(s[1] for s in x)

    return callback(add_spaces())


class CPP11Declaration(object):
    def get_id(self):
        "Return the HTML id target of this declaration"
        raise NotImplementedError()


SIMPLE_TYPE_KEYWORDS = (
    'char', 'char16_t', 'char32_t', 'wchar_t', 'short', 'int', 'long',
    'unsigned', 'signed', 'float', 'double', 'void', 'bool', 'auto',
    'const', 'volatile', 'struct', 'class', 'enum', 'union',
)

DOUBLE_OPERATOR_SYMS = (('(',')'), ('[',']'), ('>','>'))
SINGLE_OPERATOR_SYMS = (
    '+', '-', '*', '/', '%', '^', '&', '|', '~', '!', '=', '<', '>', '+=', '-=',
    '*=', '/=', '%=', '^=', '&=', '|=', '<<', '>>=', '<<=', '==', '!=', '<=',
    '&&', '||', '++', '--', ',', '->*', '->',
)

def to_tokens(lst: iter([str]), toktype) -> [Token]:
    return [Token(x, toktype) for x in lst]

def to_sym_token(sym: str) -> Token: return Token(sym, TOK_SYMBOL)
def to_ident_token(ident: str) -> Token: return Token(ident, TOK_IDENT)
def to_keyword_token(kw: str) -> Token: return Token(kw, TOK_KEYWORD)

def flatten(lst):
    res = []
    for x in lst:
        if isinstance(x, list):
            res.extend(flatten(x))
        else:
            res.append(x)
    return res

Decl = namedtuple('Decl', ['prefix', 'name', 'suffix'])

def is_cv(token: Token) -> bool:
    return token.t == TOK_KEYWORD and token.s in {'const', 'volatile', 'struct', 'class', 'enum', 'union'}

class Parser(object):
    def __init__(self, tokens):
        self._tokens = tuple(tokens)
        self._index = 0

    @property
    def rest(self) -> [Token]:
        return self._tokens[self._index:]

    @property
    def current_index(self) -> int:
        return self._index

    @property
    def empty(self) -> bool:
        return self._index >= len(self._tokens)

    def token(self) -> Token:
        try:
            retval = self._tokens[self._index]
            self._index += 1
            return retval
        except IndexError:
            raise CPP11Error("No more tokens left")

    def ident(self, identifier=None) -> str:
        tok = self.token()
        if tok.t != TOK_IDENT:
            raise CPP11Error("Identifier ({}) expected, not {} ({})".format(identifier, tok.t, tok.s))
        if identifier is not None and identifier != tok.s:
            raise CPP11Error("'{}' expected, not '{}'".format(identifier, tok.s))
        return tok.s

    def sym(self, symbol=None) -> str:
        tok = self.token()
        if tok.t != TOK_SYMBOL:
            raise CPP11Error("Symbol ({}) expected, not {} ({})".format(symbol, tok.t, tok.s))
        if symbol is not None and symbol != tok.s:
            raise CPP11Error("'{}' expected, not '{}'".format(symbol, tok.s))
        return symbol

    def one_of_sym(self, *symbols) -> str:
        tok = self.token()
        if tok.t != TOK_SYMBOL:
            raise CPP11Error("Symbol ({}) expected, not {} ({})".format(', '.join(symbols), tok.t, tok.s))
        if tok.s not in symbols:
            raise CPP11Error("One of '{}' expected, not '{}'".format(', '.join(symbols), tok.s))
        return tok.s

    def peek_sym(self, symbol) -> bool:
        old_index = self._index
        try:
            return bool(self.sym(symbol))
        except CPP11Error:
            return False
        finally:
            self._index = old_index

    def try_sym(self, symbol=None) -> str or None:
        return self.try_(lambda: self.sym(symbol))

    def try_ident(self, identifier=None) -> str or None:
        return self.try_(lambda: self.ident(identifier))

    def retreat(self, count):
        self._index -= count
        if self._index < 0:
            raise CPP11Error("Retreating too much")

    def balanced(self, left: str, right: str) -> [Token]:
        tok = self.token()
        if tok.t != TOK_SYMBOL or tok.s != left:
            return [tok]
        stack = 1
        for i, (s, t) in enumerate(self.rest):
            if t == TOK_SYMBOL:
                if s == left:
                    stack += 1
                elif s == right:
                    stack -= 1
            if stack == 0:
                end = self._index + i + 1
                retval = self._tokens[self._index-1:end]
                self._index = end
                return retval
        raise CPP11Error("Unbalanced '{}'.".format(left))

    def comma_list(self, left: str, right: str) -> [[Token]] or None:
        tok = self.try_(lambda: self.sym(left))
        if tok is None:
            return None

        retval = []
        toklist = []
        while True:
            toklist.extend(self.balanced(left, right))
            lasttok = toklist[-1]
            if lasttok.t == TOK_SYMBOL:
                is_comma = lasttok.s == ','
                is_right = lasttok.s == right
                if is_comma or is_right:
                    toklist = toklist[:-1]
                    if is_comma or toklist:
                        retval.append(toklist)
                    toklist = []
                    if is_right:
                        return retval


    def try_(self, parser) -> '*' or None:
        cur_index = self._index
        try:
            return parser()
        except CPP11Error:
            self._index = cur_index
            return None

    def keyword(self, *kws) -> str:
        tok = self.token()
        if tok.t != TOK_KEYWORD:
            raise CPP11Error("Keyword ({}) expected, not {} ({})".format(', '.join(kws), tok.t, tok.s))
        if tok.s not in kws:
            raise CPP11Error("One of '{}' expected, not '{}'".format(', '.join(kws), tok.s))
        return tok.s

    def try_keyword(self, *kws) -> str or None:
        return self.try_(lambda: self.keyword(*kws))

    def many_keywords(self, *kws) -> [str]:
        res = []
        try:
            while True:
                ok_index = self._index
                res.append(self.keyword(*kws))
        except CPP11Error:
            pass
        finally:
            self._index = ok_index
        return res

    def many_syms(self, *syms) -> [str]:
        res = []
        try:
            while True:
                ok_index = self._index
                res.append(self.one_of_sym(*syms))
        except CPP11Error:
            pass
        finally:
            self._index = ok_index
        return res

    def keyword_or_ident(self, *kws) -> str:
        token = self.token()
        if tok.t not in {TOK_KEYWORD, TOK_IDENT}:
            raise CPP11Error("Keyword or identifier ({}) expected, not {} ({})".format(', '.join(kws), tok.t, tok.s))
        if tok.s not in kws:
            raise CPP11Error("One of '{}' expected, not '{}'".format(', '.join(kws), tok.s))
        return tok.s

    def name_with_template(self) -> [Token]:
        name_token = [to_ident_token(self.ident())]
        if self.peek_sym('<'):
            template = self.balanced('<', '>')
            name_token.extend(template)
        return name_token

    def name_or_operator_name(self) -> [Token]:
        return self.either(self.name_with_template, self.operator_name)

    def namespaced_name(self) -> [Token]:
        name_part = []
        if self.try_sym('::'):
            name_part.append(to_sym_token('::'))
        name_part.extend(self.name_or_operator_name())
        while self.try_sym('::'):
            part = self.try_(self.name_or_operator_name)
            if part is not None:
                name_part.append(to_sym_token('::'))
                name_part.extend(part)
            else:
                self.retreat(1)
                break
        return name_part

    def simple_type(self, min_len=1) -> [Token]:
        res = to_tokens(self.many_keywords(*SIMPLE_TYPE_KEYWORDS), TOK_KEYWORD)
        if len(res) < min_len:
            raise CPP11Error("No simple type")
        return res

    def either(self, *parsers) -> '*':
        collected_errors = []
        for parser in parsers:
            try:
                index = self._index
                return parser()
            except CPP11Error as e:
                self._index = index
                collected_errors.append(e)
        raise CPP11Error("None matched. The collected errors are:\n - "
                            + '\n - '.join(map(str, collected_errors)))

    def sequence(self, *parsers, joiner=None) -> '*':
        if joiner is None:
            joiner = lambda lst: list(chain.from_iterable(lst))
        return joiner(parser() for parser in parsers)

    def many(self, parser) -> ['*']:
        res = []
        try:
            while True:
                ok_index = self._index
                res.append(parser())
        except CPP11Error:
            pass
        finally:
            self._index = ok_index
        return res

    def many1(self, parser) -> ['*']:
        res = self.many(parser)
        if not res:
            raise CPP11Error("None matched.")
        return res

    def decltype(self) -> [Token]:
        res = [to_keyword_token(self.keyword('decltype'))]
        res.extend(self.balanced('(', ')'))
        return res

    def get_decl_prefix(self) -> [[Token]]:
        prefix = []
        cont = True
        while cont and not self.empty:
            main_decl = self.simple_type(min_len=0)
            if not main_decl:
                main_decl = self.either(
                    self.namespaced_name,
                    self.decltype
                )
            main_decl.extend(self.simple_type(min_len=0))

            pointers = to_tokens(self.many_syms('*', '&', '&&', '::', '...'), TOK_SYMBOL)
            if pointers:
                main_decl.extend(pointers)
            else:
                cont = main_decl and all(is_cv(tok) for tok in main_decl)
            prefix.append(main_decl)

        if not prefix:
            raise CPP11Error("No type")
        return prefix

    def get_decl(self) -> Decl:
        prefix = self.get_decl_prefix()
        try:
            old_index = self._index
            self.sym('(')
            self.either(
                lambda: self.one_of_sym('*', '&', '&&'),
                lambda: self.sequence(
                    self.namespaced_name,
                    lambda: self.sym('::'),
                    lambda: self.sym('*')
                )
            )
            has_parenthesis_name = True
        except CPP11Error:
            has_parenthesis_name = False
        finally:
            self._index = old_index

        if has_parenthesis_name:
            prefix = flatten(prefix)
            parenthesis = self.balanced('(', ')')
            last_star_loc = 0
            first_paren_loc = 0
            stack = 0
            for i, (s, t) in enumerate(parenthesis):
                if t == TOK_SYMBOL:
                    if s in {'*', '&', '&&'}:
                        if stack == 0:
                            last_star_loc = i
                    elif s in {'<', '[', '{', '('}:
                        stack += 1
                    elif s in {'>', ']', '}'}:
                        stack -= 1
                    elif s == ')':
                        if stack != 0:
                            stack -= 1
                        else:
                            first_paren_loc = i
                            break
            prefix.extend(parenthesis[:last_star_loc+1])
            name = parenthesis[last_star_loc+1:first_paren_loc]
            suffix = list(parenthesis[first_paren_loc:])
        else:
            suffix = []
            name = self.try_(self.namespaced_name)
            if not name:
                name = []
                if prefix and prefix[-1]:
                    last_tok = prefix[-1][-1]
                    if (last_tok.t == TOK_IDENT
                            or (last_tok.t == TOK_SYMBOL and last_tok.s == '>')
                            or to_keyword_token('operator') in prefix[-1]
                    ):
                        name = prefix[-1]
                        prefix = prefix[:-1]
            prefix = [x for y in prefix for x in y]

        while not self.empty:
            if self.peek_sym('('):
                suffix.extend(self.balanced('(', ')'))
            elif self.peek_sym('['):
                suffix.extend(self.balanced('[', ']'))
            else:
                break

        return Decl(prefix, name, suffix)

    def operator_name(self) -> [Token]:
        res = [to_keyword_token(self.keyword('operator'))]
        new_del = self.try_keyword('new', 'delete')
        if new_del:
            res.append(to_keyword_token(new_del))
        for dos in DOUBLE_OPERATOR_SYMS:
            try:
                old_index = self._index
                self.sym(dos[0])
                self.sym(dos[1])
                res.extend(to_tokens(dos, TOK_SYMBOL))
                return res
            except CPP11Error:
                self._index = old_index

        symbol = self.try_(lambda: self.one_of_sym(*SINGLE_OPERATOR_SYMS))
        if symbol:
            res.append(to_sym_token(symbol))
            return res
        else:
            tok = self.token()
            if tok.t != TOK_LITERAL or tok.s != '""':
                raise CPP11Error("""Expected '""', not '{}'""".format(tok.s))
            res.append(tok)
            res.append(to_ident_token(self.ident()))
            return res

    def conversion_operator_name(self) -> Decl:
        self.keyword('operator')
        decl = self.get_decl()
        return Decl([], [to_keyword_token('operator')] + decl.prefix + decl.name, decl.suffix)

    # parse until reaching a symbol (including that symbol)
    def totally_balanced_until_symbols(self, *symbols) -> ([Token], str):
        stack = 0
        for i, (s, t) in enumerate(self.rest):
            if t == TOK_SYMBOL:
                if s in symbols and stack == 0:
                    end = self._index + i
                    chosen = self._tokens[self._index:end]
                    self._index = end + 1
                    return (chosen, s)
                elif s in {'[', '{', '<', '('}:
                    stack += 1
                elif s in {']', '}', '>', ')'}:
                    stack -= 1
        raise CPP11Error("Symbols '{}' not found".format(', '.join(symbols)))


class CPP11ObjDesc(ObjectDescription):
    def handle_signature(self, src, signode):
        try:
            tokens = tokenize(src)
            tokens = list(tokens)
            parser = Parser(tokens)
            parent = self.env.temp_data.get('cpp11:parent', [])
            return self.parse(parser, signode, parent)
        except CPP11Error as e:
            self.env.warn(self.env.docname, str(e), self.lineno)
            raise ValueError(e)

    def parse(self, tokens, parent: [str]) -> (str('index-name'), str('link-id')):
        raise NotImplementedError()

    def before_content(self):
        parent = self.env.temp_data.get('cpp11:parent', [])
        #self.parentized = False
        #if self.names:
        #    self.parentized = True
        parent.append(self.names[-1][2])
        self.env.temp_data['cpp11:parent'] = parent

    def after_content(self):
        #if self.parentized:
        self.env.temp_data['cpp11:parent'].pop()

    def add_target_and_index(self, link_info, src, node):
        (name, link_id, tokens) = link_info
        if link_id not in self.state.document.ids:
            node['names'].append(link_id)
            node['ids'].append(link_id)
            node['first'] = not self.names
            self.state.document.note_explicit_target(node)

            self.env.domaindata['cpp11']['objs'].setdefault(
                link_id,
                (self.env.docname, self.link_type, link_id)
            )

            link_id += '@' + self.env.docname
            node['names'].append(link_id)
            node['ids'].append(link_id)

            self.env.domaindata['cpp11']['objs'].setdefault(
                link_id,
                (self.env.docname, self.link_type, link_id)
            )

        indextext = _('%s (%s)') % (name, self.objtype)
        if indextext:
            self.indexnode['entries'].append(('single', indextext, link_id, ''))

REF_PREFIXES = {
    'macro': 'k-',
    'class': 't-',
    'struct': 't-',
    'union': 't-',
    'enum class': 't-',
    'typedef': 't-',
    'namespace': 't-',
    'function': 'f-',
    'func': 'f-',
    'data': 'f-',
    'variable': 'v-',
    'type': 't-',
    'member': 'm-',
    'property': 'f-',
    'prop': 'f-',
}

# a = struct        A = final       b = bool          B = boost
# c = char          C = char32_t    d = double        D = delete
# e =               E = extern      f = float         F = default
# g =               G =             h = short         H =
# i = int           I = ostream     j =               J =
# k = const         K = chrono      l = long          L = initializer_list
# m = unordered_map M =             n = new           N = override
# o = operator      O = ostream     p = duration      P =
# q =               Q =             r = shared_ptr    R = weak_ptr
# s = std           S = struct      t = typename      T = noexcept
# u = unique_ptr    U = unsigned    v = void          V = vector
# w = wchar_t       W = char16_t    x = variant       X = string
# y =               Y =             z =               Z =
# - = (other keyword/ident)
# : = (operators)
# _ = (literals)

REF_KNOWN_IDENTIFIERS = {
    'bool': 'b',        'char': 'c',        'char16_t': 'W',    'char32_t': 'C',
    'const': 'k',       'default': 'F',     'delete': 'D',      'double': 'd',
    'extern': 'E',      'float': 'f',       'int': 'i',         'long': 'l',
    'new': 'n',         'noexcept': 'T',    'operator': 'o',    'short': 'h',
    'signed': 'S',      'struct': 'a',      'typename': 't',    'unsigned': 'U',
    'void': 'v',        'wchar_t': 'w',

    'override': 'N',    'final': 'A',       'std': 's',         'chrono': 'K',
    'duration': 'p',    'boost': 'B',       'vector': 'V',      'unique_ptr': 'u',
    'shared_ptr': 'r',  'weak_ptr': 'R',    'string': 'X',      'ostream': 'O',
    'initializer_list': 'L',        'unordered_map': 'm',       'variant': 'x',
}

# operators must be written as 'Z???'
# a = {     A = }       b = [     B = ]         c = ,     C = :
# d = ...   D = .       e = ~     E = !         f = &     F = &&
# g = >     G = >=      h = <     H = <=        i = =     I = ==
# j = ->    J = !=      k = #     K = ##        l = /     L = /=
# m = ->*   M = .*      n =       N = &=        o = %     O = %=
# p = (     P = )       q = ?     Q = ;         r = ||    R = >>=
# s = -     S = --      t = *     T = *=        u = +     U = ++
# v = |     V = |=      w = <<    W = <<=       x = ^     X = ^=
# y =       Y =         z = +=    Z = -=        : = ::

REF_KNOWN_OPERATORS = {
    '{':'a',    '}':'A',    '[':'b',    ']':'B',    '#':'k',    '##':'K',
    '(':'p',     ')':'P',    ';':'Q',    ':':'C',    '...':'d',  '?':'q',
    '::':':',   '.':'D',    '.*':'M',   '+':'u',    '-':'s',    '*':'t',
    '/':'l',    '%':'o',    '^':'x',    '&':'f',    '|':'v',    '~':'e',
    '!':'E',    '=':'i',    '<':'h',    '>':'g',    '+=':'z',   '-=':'Z',
    '*=':'T',   '/=':'L',   '%=':'O',   '^=':'X',   '&=':'N',   '|=':'V',
    '<<':'w',   '>>=':'R',  '<<=':'W',  '==':'I',   '!=':'J',   '<=':'H',
    '>=':'G',   '&&':'F',   '||':'r',   '++':'U',   '--':'S',   ',':'c',
    '->*':'m',  '->':'j',
}

escape_re = re.compile(r'[^\-a-zA-Z0-9]')

def make_index_name_0(retval: [str], tokens: [Token]):
    for s, t in tokens:
        if t == TOK_SYMBOL:
            retval.append(':')
            retval.append(REF_KNOWN_OPERATORS[s])

        elif t == TOK_KEYWORD or t == TOK_IDENT:
            if s in REF_KNOWN_IDENTIFIERS:
                retval.append('-')
                retval.append(REF_KNOWN_IDENTIFIERS[s])
            else:
                retval.append(s)

        elif t == TOK_LITERAL:
            retval.append('_')
            retval.append(escape_re.sub(string=s, repl=lambda m:'_'+hex(ord(m.group()))[2:]))

        else:
            raise Exception("Unknown token type!")


def make_index_name(objtype: str, tokens: [Token]) -> str:
    retval = [REF_PREFIXES[objtype]]
    p = Parser(tokens)
    while not p.empty:
        opname = p.try_(p.operator_name)
        if opname:
            make_index_name_0(retval, opname)
        elif p.peek_sym('<'):
            p.balanced('<', '>')
        else:
            make_index_name_0(retval, [p.token()])

    return ''.join(retval)

def identify_potential_links(parser: Parser) -> {int: str}:
    can_link = True
    targets = {}
    active_target = []
    active_index = -2

    def perform_link():
        nonlocal targets, active_index, active_target, can_link
        if can_link:
            targets[active_index] = '::'.join(active_target)
            active_target = []
            active_index = -2
        can_link = True

    while not parser.empty:
        (s, t) = parser.token()
        if t == TOK_SYMBOL:
            if s == '>':
                break
            elif s == '<':
                targets.update(identify_potential_links(parser))
            elif s != '::':
                perform_link()

        elif t != TOK_IDENT:
            can_link = False
        elif t == TOK_IDENT and s in {'std', 'boost'}:
            can_link = False
        else:
            active_target.append(s)
            active_index = parser.current_index-1

    perform_link()
    return targets

def untokenize_with_potential_links(node, tokens):
    p = Parser(tokens)
    potential_links = identify_potential_links(p)

    def cb(it: iter([(int, str)])):
        nonlocal node
        added_something = False
        for i, s in it:
            added_something = True
            if i in potential_links:
                pnode = pending_xref('',
                                     refdomain='cpp11',
                                     reftype='type',
                                     reftarget=potential_links[i],
                                     modname=None,
                                     classname=None)
                pnode += nodes.Text(s)
                node += pnode
            else:
                node += nodes.Text(s)
        return added_something

    return untokenize(tokens, cb)

def add_parameterlist(node, parser: Parser, is_func_arg=False):
    if is_func_arg:
        parser.sym('(')
        params = desc_parameterlist()
        while True:
            # get an argument.
            (arg, terminator) = parser.totally_balanced_until_symbols(',', ')')

            is_close = terminator == ')'
            if is_close and not arg:
                break

            param = desc_parameter('', '', noemph=True)
            arg_parser = Parser(arg)
            (prefix, middle, suffix) = arg_parser.get_decl()
            prefix = untokenize_with_potential_links(param, prefix)
            middle = untokenize(middle)
            suffix.extend(arg_parser.rest)
            suffix = untokenize(suffix)
            param += nodes.Text(' ')
            param += nodes.emphasis(middle, middle)
            param += nodes.Text(suffix)
            params += param

            if is_close:
                break

        node += params
    else:
        args = parser.comma_list('(', ')')
        if args is not None:
            params = desc_parameterlist()
            for arg in args:
                content = untokenize(arg)
                params += desc_parameter(content, content)
            node += params

def prepend_parent(name_tokens: [Token], parent: [Token], objtype: str):
    if parent:
        parent_flattened = chain.from_iterable(k + [to_sym_token('::')] for k in parent)
        name_tokens = list(chain(parent_flattened, name_tokens))
    readable_name = untokenize(name_tokens)
    link_name = make_index_name(objtype, name_tokens)
    return (readable_name, link_name, name_tokens)

class CPP11MacroObjDesc(CPP11ObjDesc):
    def parse(self, parser, node, parent):
        self.objtype = 'macro'
        self.link_type = 'macro'

        node += desc_annotation('#define ', '#define ')
        name = parser.ident()
        node += desc_name(name, name)
        add_parameterlist(node, parser)
        rest = untokenize(parser.rest)
        node += nodes.Text(' ')
        node += desc_addname(rest, rest)

        return prepend_parent([to_ident_token(name)], parent, 'macro')

class CPP11EnumMemberObjDesc(CPP11ObjDesc):
    def parse(self, parser, node, parent):
        self.objtype = 'member'
        self.link_type = 'member'

        name = parser.ident()
        node += desc_name(name, name)
        if parser.peek_sym('='):
            rest = untokenize(parser.rest)
            node += nodes.Text(' ')
            node += desc_addname(rest, rest)

        return prepend_parent([to_ident_token(name)], parent, 'member')

def add_namespaced_name_to_node(node, tokens: [Token]):
    p = Parser(tokens)
    if p.try_keyword('operator'):
        rest = untokenize(tokens)
        node += desc_name(rest, rest)
        return rest

    name_index = -1
    end_name_index = -1
    template_index = -1
    while not p.empty:
        if p.try_sym('::') or p.try_sym('~'):
            pass
        elif p.peek_sym('<'):
            p.balanced('<', '>')
            template_index = p.current_index
        elif p.try_ident():
            name_index = p.current_index - 1
            end_name_index = p.current_index
            template_index = end_name_index
        else:
            opname = p.try_(p.operator_name)
            if opname:
                end_name_index = p.current_index
                name_index = end_name_index - len(opname)
                template_index = name_index
            else:
                break

    if name_index >= 0:
        pre = untokenize(tokens[:name_index])
        name = untokenize(tokens[name_index:end_name_index])
        post = untokenize(tokens[end_name_index:])
        node += desc_addname(pre, pre)
        node += desc_name(name, name)
        node += desc_addname(post, post)
        return untokenize(tokens[name_index:template_index])
    else:
        cont = untokenize(tokens)
        node += desc_addname(cont)
        return cont

TYPE_OPTIONS = [
    ('copyable', 'copyable'),
    ('noncopyable', 'non-copyable'),
    ('movable', 'movable'),
    ('nonmovable', 'non-movable'),
    ('default_constructible', 'default-constructible'),
    ('non_default_constructible', 'non-default-constructible'),
    ('pod', 'POD'),
    ('ostream', 'has ostream<<'),
]

class CPP11TypeObjDesc(CPP11ObjDesc):
    option_spec = dict(noindex=directives.flag, **{k: directives.flag for k, _ in TYPE_OPTIONS})

    def parse(self, parser, node, parent):
        self.link_type = 'type'
        if parser.try_keyword('protected'):
            node += desc_annotation('protected ', 'protected ');
        if parser.try_ident('type'):
            name = self.parse_type(parser, node)
        else:
            name = self.parse_normal(parser, node)

        extra_nodes = []
        for opt, txt in TYPE_OPTIONS:
            if opt in self.options:
                should_append_extra_node = True
                content = nodes.list_item('', nodes.paragraph(txt, txt), classes=[opt])
                extra_nodes.append(content)
        if extra_nodes:
            node += nodes.bullet_list('', *extra_nodes, classes=['type-compact-list'])
        return prepend_parent(name, parent, self.objtype)

    def parse_type(self, parser, node):
        self.objtype = 'type'

        node += desc_annotation('type ', 'type ')

        name = parser.namespaced_name()
        link_name = make_index_name(self.objtype, name)
        add_namespaced_name_to_node(node, name)

        if parser.try_sym('='):
            node += nodes.Text(' = ')
            addname = desc_addname('', '')
            untokenize_with_potential_links(addname, parser.rest)
            node += addname
        return name


    def parse_normal(self, parser, node):
        self.objtype = parser.keyword('struct', 'class', 'enum', 'union', 'namespace')
        if self.objtype == 'enum':
            parser.try_keyword('struct', 'class')
            self.objtype = 'enum class'

        node += desc_annotation(self.objtype, self.objtype)
        node += nodes.Text(' ')

        name = parser.namespaced_name()
        add_namespaced_name_to_node(node, name)

        if parser.try_ident('final'):
            node += desc_annotation(' final', ' final')
        if parser.try_sym(':'):
            node += nodes.Text(' : ')
            while True:
                protections = parser.many_keywords('virtual', 'public', 'protected', 'private')
                prot_str = ' '.join(protections)
                node += desc_annotation(prot_str, prot_str)
                node += nodes.Text(' ')
                base = parser.either(parser.namespaced_name, parser.simple_type)
                base_disp = untokenize(base)
                node += desc_addname(base_disp, base_disp)
                if parser.try_sym(','):
                    node += nodes.Text(', ')
                else:
                    break

        return name

class CPP11FunctionObjDesc(CPP11ObjDesc):
    def parse(self, parser, node, parent):
        self.objtype = self.link_type = self.name.split(':')[-1]

        # is_function = self.objtype == 'function'
        if self.objtype == 'property':
            prop_access = parser.ident()
            prop_display = 'declprop::' + prop_access
        else:
            prop_display = None

        func_attribs = parser.many_keywords(
            'register', 'static', 'thread_local', 'extern', 'mutable', 'inline',
            'virtual', 'explicit', 'constexpr', 'public', 'private', 'protected'
        )

        try:
            (prefix, name, suffix) = parser.either(
                parser.conversion_operator_name,
                parser.get_decl
            )
        except CPP11Error as e:
            #print(untokenize(parser.rest), e)
            #raise e
            # try c'tor / d'tor name
            prefix = []
            name = [to_sym_token('~')] if parser.try_sym('~') else []
            name.append(to_ident_token(parser.ident()))
            suffix = parser.balanced('(', ')')

        really_suffix = parser.rest

        has_prefix = False

        if func_attribs:
            func_attribs_ann = untokenize(to_tokens(func_attribs, TOK_KEYWORD))
            node += desc_annotation(func_attribs_ann, func_attribs_ann)
            node += nodes.Text(' ')
            has_prefix = True

        if prop_display:
            node += desc_annotation(prop_display, prop_display)
            node += termsep()

        addname = desc_addname('', '')
        has_prefix = untokenize_with_potential_links(addname, prefix)
        if has_prefix:
            node += addname

        if self.objtype == 'function':
            if has_prefix:
                node += termsep()
        else:
            node += nodes.Text(' ')
        # ^ force functions splitting into 2 lines to make the function name
        #   easier to see.

        add_namespaced_name_to_node(node, name)

        params_parser = Parser(suffix)
        close_parens = untokenize(params_parser.many_syms(')'))
        if close_parens:
            node += desc_addname(close_parens, close_parens)

        if self.objtype == 'function':
            add_parameterlist(node, params_parser, is_func_arg=True)

        suffix = untokenize(params_parser.rest)
        if suffix:
            node += desc_addname(suffix, suffix)

        really_suffix_ann = untokenize(really_suffix)
        if really_suffix_ann:
            node += nodes.Text(' ')
            node += desc_annotation(really_suffix_ann, really_suffix_ann)

        return prepend_parent(name, parent, 'function')

class Dummy(object):
    def __iadd__(self, other):
        return self

def tokenize_with_explicit_doc(src: str, src_doc_name: str) -> (str('doc_name'), str('fixed_target'), iter([Token])):
    try:
        at_index = src.index('@')
        explicit_doc = src[at_index:]
        tokens_src = src[:at_index]
        if explicit_doc == '@here':
            explicit_doc = '@' + src_doc_name
            src = tokens_src + explicit_doc
    except ValueError:
        explicit_doc = ''
        tokens_src = src

    return (explicit_doc, src, tokenize(tokens_src))

class CPP11XRefRole(XRefRole):
    def process_link(self, env, refnode, has_explicit_title, title, target):
        if not has_explicit_title:
            if target.startswith('~'):
                target = target[1:]
                (_, target, title_tokens) = tokenize_with_explicit_doc(target, env.docname)
                title_tokens = list(title_tokens)
                real_name = add_namespaced_name_to_node(Dummy(), title_tokens)
                title = real_name
                if self.fix_parens:
                    (title, target) = self._fix_parens(env, has_explicit_title, title, target)

        return (title, target)

class CPP11Domain(Domain):
    name = 'cpp11'
    label = 'C++11'

    object_types = {
        'type': ObjType(l_('type'), 'type'),
        'function': ObjType(l_('function'), 'func', 'data'),
        'data': ObjType(l_('data'), 'func', 'data'),
        'member': ObjType(l_('member'), 'member'),
        'macro': ObjType(l_('macro'), 'macro'),
        'property': ObjType(l_('property'), 'prop'),
    }

    directives = {
        'type': CPP11TypeObjDesc,
        'function': CPP11FunctionObjDesc,
        'data': CPP11FunctionObjDesc,
        'member': CPP11EnumMemberObjDesc,
        'macro': CPP11MacroObjDesc,
        'property': CPP11FunctionObjDesc,
    }

    roles = {
        'type': CPP11XRefRole(),
        'member': CPP11XRefRole(),
        'macro': CPP11XRefRole(),
        'func': CPP11XRefRole(fix_parens=True),
        'data': CPP11XRefRole(),
        'prop': CPP11XRefRole(),
    }

    initial_data = {
        'objs': {}
    }

    def clear_doc(self, docname):
        for fullname, (fn, _, _) in list(self.data['objs'].items()):
            if fn == docname:
                del self.data['objs'][fullname]

    def resolve_xref(self, env, src_docname, builder, link_role, target_name, node, cont_node):
        (explicit_doc, _, target_tokens) = tokenize_with_explicit_doc(target_name, src_docname)
        parsed_target_name = make_index_name(link_role, target_tokens) + explicit_doc
        if parsed_target_name not in self.data['objs']:
            #print(target_name, self.data['objs'], sep='\n')
            #raise Exception(target_name + " not found 670")
            return None
        (target_docname, target_role, link_id) = self.data['objs'][parsed_target_name]
        if target_role not in self.objtypes_for_role(link_role):
            #print(target_role, link_role, sep='\n')
            #raise Exception(target_name + " not found 675")
            return None
        return make_refnode(builder, src_docname, target_docname, link_id, cont_node, target_name)

    def get_objects(self):
        for refname, (docname, typ, _) in self.data['objs'].items():
            yield (refname, refname, typ, docname, refname, 1)

def setup(app):
    app.add_domain(CPP11Domain)

