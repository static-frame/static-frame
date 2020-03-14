

For each container, the complete public API is presented below. Note that interface endpoints, such as ``iter_element``, are expanded to show all interface sub components, such as ``apply`` and ``map_any``.

This is an overview for quick reference; for detailed documentation, start with :ref:`structures`.


.. jinja:: ctx

    {% for name, cls in interface %}

    {{ name }}
    -------------------------------------------------


    {% for group, frame_sub in cls.interface.iter_group_items('group', axis=0) %}

    {{ name }}: {{ group }}
    ..........................................................

    .. csv-table::
        :header-rows: 0

        {% for label, row in frame_sub.iter_tuple_items(axis=1) -%}
            {% if label == "[]" -%}
            ":obj:`static_frame.{{name}}{{label}}`", "{{row.doc}}"
            {% else -%}
            ":obj:`static_frame.{{name}}.{{label}}`", "{{row.doc}}"
            {% endif -%}
        {% endfor %}

    {% endfor %}

    {% endfor %}




