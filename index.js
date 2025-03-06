<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Formulario de Sustitución con Nunjucks</title>
    <link rel="stylesheet" href="styles.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/nunjucks/3.2.3/nunjucks.min.js"></script>
</head>
<body>
    <h1>Formulario de Sustitución de Variables</h1>

    <!-- Formulario dinámico -->
    <form id="substitutionForm">
        <table>
            <thead>
                <tr>
                    <th>Clave</th>
                    <th>Valor</th>
                </tr>
            </thead>
            <tbody id="formTableBody">
                <!-- Las filas se generarán dinámicamente -->
            </tbody>
        </table>
        <button type="button" id="generateButton">Generar Banner</button>
    </form>

    <!-- Contenedor para mostrar el banner generado -->
    <h2>Banner Generado:</h2>
    <div id="bannerPreview"></div>

    <script src="app.js"></script>
</body>
</html>
