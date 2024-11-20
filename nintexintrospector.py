import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple
import csv


class NintexPromptGenerator:
    def __init__(self, xml_file_path: str):
        with open(xml_file_path, 'r', encoding='utf-8') as file:
            self.xml_content = file.read()
        self.prompts_by_activity = {
            'AssignVariable': self._generate_variable_prompt,
            'SendEmail': self._generate_email_prompt,
            'FlexibleTask': self._generate_task_prompt,
            'IfElse': self._generate_condition_prompt,
            'UpdateItemMetadata': self._generate_metadata_prompt,
            'CheckIn': self._generate_checkin_prompt,
            'CreateListItem': self._generate_list_item_prompt
        }

    def _activity_to_xml(self, activity: ET.Element) -> str:
        return ET.tostring(activity, encoding='unicode', method='xml')

    def generate_individual_prompts(self) -> List[Dict]:
        root = ET.fromstring(self.xml_content)
        ns = {'nw': 'http://schemas.nintex.com/workflow/2010'}
        activities = []

        for idx, activity in enumerate(root.findall('.//nw:Activity', ns)):
            activity_type = activity.get('Type')
            description = activity.get('Description', '')
            properties = self._extract_properties(activity)
            xml_content = self._activity_to_xml(activity)

            if activity_type in self.prompts_by_activity:
                prompt = self.prompts_by_activity[activity_type](properties, xml_content)
                activities.append({
                    'id': f'activity_{idx}',
                    'type': activity_type,
                    'description': description,
                    'properties': properties,
                    'xml': xml_content,
                    'prompt': prompt
                })

        return activities

    def _extract_properties(self, activity: ET.Element) -> Dict:
        properties = {}
        for prop in activity.findall('.//Property'):
            name = prop.get('Name', '')
            value = prop.text or ''
            properties[name] = value
        return properties

    def _generate_variable_prompt(self, props: Dict, xml: str) -> str:
        return f"""
        Convierte esta actividad de asignación de variable de Nintex a Power Automate:

        XML de la actividad:
        {xml}

        Detalles:
        Variable: {props.get('VariableName')}
        Valor: {props.get('Value')}

        Requisitos:
        1. Usar Initialize variable o Set variable según corresponda
        2. Mantener el tipo de dato original
        3. Generar un JSON válido para importar en Power Automate
        4. Usar la sintaxis correcta para referencias dinámicas
        """

    def _generate_email_prompt(self, props: Dict, xml: str) -> str:
        return f"""
        Convierte esta actividad de envío de correo de Nintex a Power Automate:

        XML de la actividad:
        {xml}

        Detalles:
        Para: {props.get('To')}
        Asunto: {props.get('Subject')}
        Cuerpo: {props.get('Body')}

        Requisitos:
        1. Usar el conector Office 365 Outlook
        2. Mantener el formato HTML si existe
        3. Generar un JSON válido para importar en Power Automate
        4. Incluir manejo de adjuntos si existen
        """

    def _generate_task_prompt(self, props: Dict, xml: str) -> str:
        return f"""
        Convierte esta tarea de aprobación de Nintex a Power Automate:

        XML de la actividad:
        {xml}

        Detalles:
        Asignado a: {props.get('AssignTo')}
        Fecha límite: {props.get('DueDate')}

        Requisitos:
        1. Usar el conector de Aprobaciones moderno
        2. Configurar el tiempo de espera
        3. Generar un JSON válido para importar en Power Automate
        4. Incluir manejo de respuestas múltiples
        """

    def _generate_condition_prompt(self, props: Dict, xml: str) -> str:
        return f"""
        Convierte esta condición de Nintex a Power Automate:

        XML de la actividad:
        {xml}

        Detalles:
        Condición: {props.get('Condition')}

        Requisitos:
        1. Usar Condition o Switch según corresponda
        2. Mantener la lógica exacta de evaluación
        3. Generar un JSON válido para importar en Power Automate
        """

    def _generate_metadata_prompt(self, props: Dict, xml: str) -> str:
        return f"""
        Convierte esta actividad de actualización de metadatos de Nintex a Power Automate:

        XML de la actividad:
        {xml}

        Detalles:
        Estado: {props.get('Status')}

        Requisitos:
        1. Usar el conector de SharePoint
        2. Actualizar los metadatos del elemento
        3. Generar un JSON válido para importar en Power Automate
        """

    def _generate_checkin_prompt(self, props: Dict, xml: str) -> str:
        return f"""
        Convierte esta actividad de Check-in de Nintex a Power Automate:

        XML de la actividad:
        {xml}

        Detalles:
        Comentarios: {props.get('Comments')}

        Requisitos:
        1. Usar el conector de SharePoint
        2. Realizar el check-in del documento
        3. Generar un JSON válido para importar en Power Automate
        """

    def _generate_list_item_prompt(self, props: Dict, xml: str) -> str:
        return f"""
        Convierte esta actividad de creación de elemento de lista de Nintex a Power Automate:

        XML de la actividad:
        {xml}

        Detalles:
        Lista: {props.get('ListName')}

        Requisitos:
        1. Usar el conector de SharePoint
        2. Crear el elemento en la lista especificada
        3. Generar un JSON válido para importar en Power Automate
        """

    def save_prompts_to_csv(self, activities: List[Dict], output_file: str):
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['id', 'type', 'description', 'xml', 'prompt']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for activity in activities:
                writer.writerow({
                    'id': activity['id'],
                    'type': activity['type'],
                    'description': activity['description'],
                    'xml': activity['xml'],
                    'prompt': activity['prompt']
                })


if __name__ == "__main__":
    # Ruta al archivo XML de Nintex
    xml_file_path = "workflow_nintex.xml"

    try:
        # Crear instancia del generador de prompts
        generator = NintexPromptGenerator(xml_file_path)

        # Generar los prompts
        activities = generator.generate_individual_prompts()

        # Guardar en CSV
        generator.save_prompts_to_csv(activities, "output_prompts.csv")

        # Procesar cada actividad individualmente
        for activity in activities:
            print(f"\n--- Actividad: {activity['id']} ---")
            print(f"Tipo: {activity['type']}")
            print(f"Descripción: {activity['description']}")
            print("XML de la actividad:")
            print(activity['xml'])
            print("Prompt para LLM:")
            print(activity['prompt'])

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo XML en la ruta: {xml_file_path}")
    except ET.ParseError as e:
        print(f"Error al parsear el XML: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")
