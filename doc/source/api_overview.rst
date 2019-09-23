

.. jinja:: ctx

    {% for name, frame in interface %}

    {{ name }}
    ====================================================


    {% for group, frame_sub in frame.iter_group_items('group', axis=0) %}

    {{ group }}
    -------------------------------------------------------

    .. csv-table::
        :header-rows: 0

        {% for label, row in frame_sub.iter_tuple_items(axis=1) -%}
            ``{{label}}``, "{{row.doc}}"
        {% endfor %}

    {% endfor %}

    {% endfor %}



# :py:meth:`static_frame.{{name}}.{{label}}`, "{{row.doc}}"
