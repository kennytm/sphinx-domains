cpp11.py
========

This is a `Sphinx domain`_ written for my C++11 projects, because the default
C++ domain does not support C++11.

Usage
-----

Copy the ``cpp11.py`` file to your Sphinx sources folder, and then edit the
``conf.py`` with the following changes:

1. Add ``'cpp11'`` to the ``extensions`` array.
2. Set ``cpp11`` as the ``primary_domain``, e.g.::

    extensions = [..., 'cpp11']
    ...
    primary_domain = 'cpp11'
    highlight_language = 'c++'

3. Copy the ``cpp11.css`` and ``decorations.png`` to the ``_static`` folder.
4. ``@import url(cpp11.css);`` inside your stylesheet (usually
   ``_static/basic.css``).

Directives
----------

cpp11:type
~~~~~~~~~~

The "type" directive declares a type. Unlike the default C++ domain, you need to
specify what type you are declaring, e.g.::

    .. cpp11:type:: type size_t = unsigned long

        This is a typedef.

    .. cpp11:type:: enum class std::errc

        This is a strongly-typed enum.

    .. cpp11:type:: class Foo final : public Bar

        This is a final derived class.

        .. cpp11:type:: protected union Variant

            This is a protected inner union.

    .. cpp11:type:: struct std::vector<T>

        This is a templated struct.

Types can have some associated options to describe the type::

    .. cpp11:type:: struct Vector3
        :copyable:
        :movable:
        :default_constructible:
        :pod:
        :ostream:

        ...

    .. cpp11:type:: class Singleton
        :noncopyable:
        :nonmovable:
        :non_default_constructible:

        ...

cpp11:function
~~~~~~~~~~~~~~

::

    .. cpp11:function:: void std::sort<It, Comp>(It begin, It end, Comp comp)

This directive declares a function or method. Because C++'s return type can be
very verbose, the translated method name will be shown a line below the return
type, like:

    *static inline* ``std::unordered_map<std::string, std::shared_ptr<int>>``

    ``Widgets::``\ **get_mapping**\(std::string *domain*) *noexcept*

cpp11:data
~~~~~~~~~~

::

    .. cpp11:data:: extern std::ostream std::cout

        Console out

    .. cpp11:data:: static constexpr float Math::pi = 3.14159265f

        π

cpp11:member
~~~~~~~~~~~~

::

    .. cpp11:type:: enum class Foo

        Foo

        .. cpp11:member:: first_member

            The first member

        .. cpp11:member:: another_member = 42

            Another member

This directive declares an enum member.

cpp11:macro
~~~~~~~~~~~

::

    .. cpp11:macro:: CONCAT(foo, bar) foo##bar

        A macro to concatenate two tokens.

cpp11:property
~~~~~~~~~~~~~~

This directive declares a property using the `<utils/property.hpp>`_ module.

::

    .. cpp11:property:: read_write_byval float font_size

        The font size.

Roles
-----

The following roles are available:

* ``:type:``
* ``:member:``
* ``:macro:``
* ``:func:``
* ``:data:``
* ``:prop:``

The link of the role must be a fully-qualified name, e.g.
``:type:`std::string```. As usual, prefix a ``~`` to avoid showing the namespace
parts e.g. ``:func:`~std::vector<T>::at```.

Sometimes the same name is defined across different files. To disambiguate, you
could use ``@`` to specify which file to link to, e.g.::

    This is the "utility" module. We have the :func:`std::move@here` method.
    There is also another :func:`std::move@std/algorithm` in "algorithm".

.. _Sphinx domain: http://sphinx.pocoo.org/latest/domains.html
.. _<utils/property.hpp>: https://github.com/kennytm/utils/blob/master/property.hpp

