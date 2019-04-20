

from functools import partial

VERSION_BOOTSTRAP = '4.3.1'
VERSION_DATATABLES = '1.10.19'
VERSION_JQUERY = '3.3.1'
VERSION_JQUERY_UI = '1.12.0'

TEMPLATE = partial('''
<!doctype html>
<html>
<head>
    <title>StaticFrame</title>

    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta charset="UTF-8">

    <link rel="stylesheet"
        href="https://maxcdn.bootstrapcdn.com/bootstrap/{version_bootstrap}/css/bootstrap.min.css" />
    <link rel="stylesheet"
        href="https://cdn.datatables.net/{version_datatables}/css/jquery.dataTables.min.css" />

    <script src="https://code.jquery.com/jquery-{version_jquery}.min.js"></script>
    <script src="https://code.jquery.com/ui/{version_jquery_ui}/jquery-ui.min.js"></script>
    <script src="https://cdn.datatables.net/{version_datatables}/js/jquery.dataTables.min.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/{version_bootstrap}/js/bootstrap.min.js"></script>

    <script>
        $(document).ready(function() {{
            $("#{}").DataTable();
            }});
    </script>
    <style>
        body {{
            padding-top: 50px;
            overflow-y:scroll;
        }}
        table td, th {{
            font-size: 80%;
        }}
        table th {{
            background: #dddddd;
        }}
    </style>
</head>

<body>
<div class="container-fluid body-main">
{}
</div>
</body>
</html>
'''.format, version_bootstrap=VERSION_BOOTSTRAP,
        version_datatables=VERSION_DATATABLES,
        version_jquery=VERSION_JQUERY,
        version_jquery_ui=VERSION_JQUERY_UI)
