.. NOTE: auto-generated file, do not edit

.. jinja:: ctx

    {# api_detail /////////////////////////////////////////////////////////////// #}
    {% macro api_detail(name, cls, frame_items, examples_defined) -%}
    
    .. _api-detail-{{ name }}:
    
    {{ name }}
    ================================================================================
    
    Overview: :ref:`api-overview-{{ name }}`
    
    {# docs are on __init__, not the class, so this just provides an anchor for obj refs #}
    .. autoclass:: static_frame.{{cls.__name__}}
    
    {% for group, frame_sub in frame_items %}
    
    .. _api-detail-{{ name }}-{{ group }}:
    
    {{ name }}: {{ group }}
    ................................................................................
    
    Overview: :ref:`api-overview-{{ name }}-{{ group }}`
    
    {% for signature, row in frame_sub.iter_tuple_items(axis=1) -%}
    
    {# anchor for linking from overview #}
    .. _api-sig-{{ name }}-{{ row.signature_no_args }}:
    
    
    {% if row.use_signature and signature.startswith('[') %}
    
    .. py:method:: {{ name }}{{ signature }}  {# NOTE: no dot! #}
    
    {% elif row.use_signature and signature.startswith('interface') %}
    
    .. py:attribute:: {{ name }}.{{ signature }}
    
        {{ row.doc }}
    
    {% elif row.use_signature and not row.is_attr %}
    
    .. py:method:: {{ name }}.{{ signature }}
        :noindex:
    
    {% elif row.use_signature and row.is_attr %}
    
    .. py:attribute:: {{ name }}.{{ signature }}
        :noindex:
    
    {% elif group == 'Attribute' or signature == 'values' or row.is_attr %}
    
    .. autoattribute:: static_frame.{{ row.reference }}
        :noindex:
    
    {% else %}
    
    .. automethod:: static_frame.{{ row.reference }}
        :noindex:
    
    {% endif %}
    
    
    {# if a signature has been used, then we need to augment with doc with reference #}
    {% if row.use_signature %}
    
        {% if row.reference and row.is_attr %}
    
        .. autoattribute:: static_frame.{{ row.reference }}
            :noindex:
    
        {% elif row.reference %}
    
        .. automethod:: static_frame.{{ row.reference }}
            :noindex:
    
        {% endif %}
    
    {% endif %}
    
    
    {# if delegate_reference is defined, always include it #}
    {% if row.delegate_reference %}
    
        {% if row.delegate_is_attr %}
    
        .. autoattribute:: static_frame.{{ row.delegate_reference }}
            :noindex:
    
        {% else %}
    
        .. automethod:: static_frame.{{ row.delegate_reference }}
            :noindex:
    
        {% endif %}
    
    {% endif %}
    
    
    {# example ////////////////////////////////////////////////////////////////// #}
    {# for debugging: ``start_{{ name }}-{{ row.signature_no_args }}`` #}
    
    {% if name + '-' + row.signature_no_args in examples_defined %}
    
        .. literalinclude:: ../../../static_frame/test/unit/test_doc.py
           :language: python
           :start-after: start_{{ name }}-{{ row.signature_no_args }}
           :end-before: end_{{ name }}-{{ row.signature_no_args }}
    
    {% endif %}
    
    {% endfor %}
    
    :ref:`api-detail-{{ name }}`: {% for group_xref, _ in frame_items %}:ref:`{{ group_xref }}<api-detail-{{ name }}-{{ group_xref }}>`{{ " | " if not loop.last }}{% endfor %}
    
    {% endfor %}
    {%- endmacro %}
    
    {# api_overview ///////////////////////////////////////////////////////////// #}
    {% macro api_overview(name, cls, frame_items, examples_defined) -%}
    
    .. _api-overview-{{ name }}:
    
    {{ name }}
    ================================================================================
    
    Detail: :ref:`api-detail-{{ name }}`
    
    
    {% for group, frame_sub in frame_items %}
    
    .. _api-overview-{{ name }}-{{ group }}:
    
    {{ name }}: {{ group }}
    --------------------------------------------------------------------------------
    
    Detail: :ref:`api-detail-{{ name }}-{{ group }}`
    
    .. csv-table::
        :header-rows: 0
    
        {% for signature, row in frame_sub.iter_tuple_items(axis=1) -%}
            {% if signature.startswith('[') -%}
            :ref:`Detail<api-sig-{{ name }}-{{ row.signature_no_args }}>`, ":obj:`{{name}}{{signature}}`", "{{row.doc}}"
            {% else -%}
            :ref:`Detail<api-sig-{{ name }}-{{ row.signature_no_args }}>`, ":obj:`{{name}}.{{signature}}`", "{{row.doc}}"
            {% endif -%}
        {% endfor %}
    
    :ref:`api-overview-{{ name }}`: {% for group_xref, _ in frame_items %}:ref:`{{ group_xref }}<api-overview-{{ name }}-{{ group_xref }}>`{{ " | " if not loop.last }}{% endfor %}
    
    {% endfor %}
    {%- endmacro %}

    {{ api_overview(examples_defined=examples_defined, *interface['IndexGO']) }}
