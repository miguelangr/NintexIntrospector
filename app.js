document.addEventListener("DOMContentLoaded", () => {
    // Configurar Nunjucks
    nunjucks.configure({ autoescape: true });

    // Template base (puedes ponerlo en un archivo aparte)
    const bannerTemplate = `
    <div class="private" style="
        display: flex;
        max-width: {{ max_width }};
        max-height: {{ max_height }};
        overflow: hidden;
        font-family: {{ font_family }};
        border: {{ border }};
        border-radius: {{ border_radius }};
        box-shadow: {{ box_shadow }};
    ">
        <img class="photo" 
             src="{{ image_src }}" 
             alt="{{ image_alt }}" 
             style="
                 width: {{ photo_width }};
                 object-fit: {{ photo_object_fit }};
             " />
        <div class="text" style="
            width: {{ text_width }};
            padding: {{ text_padding }};
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            box-sizing: border-box;
        ">
            <p class="title" style="
                margin: 0;
                width: 100%;
                font-size: {{ title_font_size }};
            ">
                {{ title_content }}
            </p>
            <p class="caption" style="
                margin: 0;
                width: 100%;
                font-size: {{ caption_font_size }};
            ">
                {{ caption_content }}
            </p>
        </div>
    </div>
    `;

    // Tabla inicial con claves y valores por defecto
    const defaultData = [
        { key: "max_width", value: "1200px" },
        { key: "max_height", value: "250px" },
        { key: "font_family", value: "Arial, sans-serif" },
        { key: "border", value: "1px solid #000" },
        { key: "border_radius", value: "5px" },
        { key: "box_shadow", value: "0 0 10px rgba(0, 0, 0, 0.1)" },
        { key: "image_src", value: "/private/background.jpg" },
        { key: "image_alt", value: "Private Banking" },
        { key: "photo_width", value: "25%" },
        { key: "photo_object_fit", value: "cover" },
        { key: "text_width", value: "75%" },
        { key: "text_padding", value: "20px" },
        { key: "title_font_size", value: "1.5em" },
        { key: "title_content", value: "Welcome to Private Banking" },
        { key: "caption_font_size", value: "1em" },
        { key: "caption_content", value: "Discover amazing products at our Private Banking" }
    ];

    const formTableBody = document.getElementById("formTableBody");
    const generateButton = document.getElementById("generateButton");
    const bannerPreview = document.getElementById("bannerPreview");

    // Función para crear una fila dinámica
    function createRow(key, defaultValue) {
        const row = document.createElement("tr");

        // Columna de la clave (solo lectura)
        const keyCell = document.createElement("td");
        keyCell.textContent = key;

        // Columna del valor (editable)
        const valueCell = document.createElement("td");
        const input = document.createElement("input");
        input.type = "text";
        input.placeholder = defaultValue; // Muestra el valor por defecto en gris
        input.value = defaultValue; // Valor inicial
        valueCell.appendChild(input);

        row.appendChild(keyCell);
        row.appendChild(valueCell);
        return row;
    }

    // Generar las filas iniciales del formulario
    defaultData.forEach(({ key, value }) => {
        const row = createRow(key, value);
        formTableBody.appendChild(row);
    });

    // Evento para generar el banner final
    generateButton.addEventListener("click", () => {
        const data = {}; // Objeto para almacenar los valores
        const rows = formTableBody.querySelectorAll("tr");

        rows.forEach(row => {
            const key = row.children[0].textContent; // Clave (columna 1)
            const inputValue = row.children[1].querySelector("input").value; // Valor ingresado (columna 2)
            data[key] = inputValue || row.children[1].querySelector("input").placeholder; // Usa el valor o el placeholder
        });

        // Renderizar el template con Nunjucks
        const renderedHTML = nunjucks.renderString(bannerTemplate, data);

        // Mostrar el banner generado
        bannerPreview.innerHTML = renderedHTML;
    });
});
