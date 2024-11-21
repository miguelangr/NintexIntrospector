import xml.etree.ElementTree as ET
from ctransformers import AutoModelForCausalLM
import json
from typing import Dict, List, Any
from pathlib import Path
import os

class NintexToPowerAutomate:
    def __init__(self, model_path: str):
        """
        Inicializa el procesador con el modelo local
        """
        print(f"Inicializando modelo desde: {model_path}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="mistral",
            context_length=2048,
            max_new_tokens=512,
            temperature=0.1
        )
        self.action_tree = {}
        self.processed_actions = {}
        self.parent_map = {}

    def generate_response(self, prompt: str) -> str:
        """
        Genera una respuesta usando el modelo
        """
        try:
            response = self.llm(
                prompt,
                max_new_tokens=512,
                temperature=0.1,
                stop=["}"]
            )
            return response.strip()
        except Exception as e:
            print(f"Error en la generación de respuesta: {str(e)}")
            return ""

    def load_workflow(self, xml_path: str) -> ET.Element:
        """
        Carga el archivo XML del workflow de forma permisiva
        """
        try:
            print(f"Cargando workflow desde: {xml_path}")
            
            # Verificar si el archivo existe
            if not os.path.exists(xml_path):
                print(f"ERROR: El archivo {xml_path} no existe")
                return None
                
            # Leer el contenido del archivo
            with open(xml_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()
                print(f"Contenido XML leído, tamaño: {len(xml_content)} bytes")
            
            parser = ET.XMLParser(encoding="utf-8")
            tree = ET.parse(xml_path, parser=parser)
            root = tree.getroot()
            
            print(f"Root tag: {root.tag}")
            
            # Verificar si hay acciones y mostrar detalles
            actions = root.findall('.//NWActionConfig')
            if len(actions) == 0:
                print("ADVERTENCIA: No se encontraron acciones NWActionConfig en el XML")
            else:
                print(f"Se encontraron {len(actions)} acciones")
                for idx, action in enumerate(actions, 1):
                    action_id = action.get('id', 'Sin ID')
                    action_name = action.get('name', 'Sin nombre')
                    print(f"Acción {idx}: ID={action_id}, Nombre={action_name}")
            
            # Crear mapa de padres para toda la estructura
            self.parent_map = {c: p for p in root.iter() for c in p}
            print(f"Mapa de padres creado con {len(self.parent_map)} relaciones")
            
            return root
            
        except ET.ParseError as e:
            print(f"Error al parsear XML: {str(e)}")
            print("Contenido problemático:")
            print(xml_content[:500] + "..." if len(xml_content) > 500 else xml_content)
            return None
        except Exception as e:
            print(f"Error inesperado al cargar XML: {str(e)}")
            return None

    def extract_action_properties(self, action: ET.Element) -> Dict[str, Any]:
        """
        Extrae todas las propiedades de una acción de forma segura
        """
        properties = {}
        try:
            print(f"Extrayendo propiedades para acción: {action.get('name', 'Sin nombre')}")
            
            # Mostrar todos los atributos de la acción
            print(f"Atributos de la acción: {action.attrib}")
            
            for prop in action.findall('.//property'):
                prop_name = prop.get('name', '')
                prop_value = prop.text or ''
                if prop_name:
                    properties[prop_name] = prop_value
                    print(f"Propiedad encontrada: {prop_name} = {prop_value}")
            
            action_data = {
                'id': action.get('id', ''),
                'name': action.get('name', ''),
                'type': action.get('type', ''),
                'properties': properties
            }
            print(f"Datos extraídos: {json.dumps(action_data, indent=2)}")
            return action_data
            
        except Exception as e:
            print(f"Error al extraer propiedades: {str(e)}")
            return {
                'id': action.get('id', ''),
                'error': str(e)
            }

    def process_single_action(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa una única acción usando el modelo
        """
        try:
            print(f"Procesando acción: {action_data.get('name', 'Sin nombre')}")
            
            prompt = f"""
            Analiza esta acción de Nintex Workflow y genera su equivalente en Power Automate.
            Genera solo el JSON de la acción equivalente en Power Automate.
            
            Acción Nintex:
            {json.dumps(action_data, indent=2)}
            
            Respuesta en formato JSON:
            {{
            """

            print("Generando respuesta con el modelo...")
            response = self.generate_response(prompt)
            print(f"Respuesta generada, longitud: {len(response)}")
            
            # Asegurarse de que tenemos un JSON válido
            json_str = "{" + response.split("{", 1)[-1]
            if not json_str.endswith("}"):
                json_str += "}"
            
            try:
                power_automate_action = json.loads(json_str)
                print("JSON válido generado")
            except json.JSONDecodeError:
                print("Error en el primer intento de parse JSON, intentando limpiar...")
                # Si falla, intentar limpiar el JSON
                json_str = json_str.replace('\n', '').replace('\r', '')
                power_automate_action = json.loads(json_str)
                print("JSON válido generado después de limpieza")
                
        except Exception as e:
            print(f"Error en process_single_action: {str(e)}")
            power_automate_action = {
                "error": f"No se pudo generar una traducción válida: {str(e)}",
                "original_action": action_data
            }

        return power_automate_action

    def find_parent_action(self, action: ET.Element) -> str:
        """
        Encuentra el ID del padre de forma segura usando el mapa de padres
        """
        try:
            parent = self.parent_map.get(action)
            if parent is not None:
                print(f"Padre encontrado para {action.get('id', 'Sin ID')}: {parent.tag}")
                if parent.tag == 'NWActionConfig':
                    return parent.get('id', '')
            return 'root'
        except Exception as e:
            print(f"Error al buscar padre: {str(e)}")
            return 'root'

    def build_action_tree(self, root: ET.Element) -> Dict[str, List[Dict[str, Any]]]:
        """
        Construye el árbol de acciones de forma segura
        """
        print("\nIniciando procesamiento del árbol de acciones...")
        
        try:
            # Primero, encontrar todas las acciones
            actions = root.findall('.//NWActionConfig')
            print(f"Total de acciones encontradas: {len(actions)}")
            
            for action in actions:
                action_id = action.get('id', '')
                print(f"\nProcesando acción: {action_id}")
                
                parent_id = self.find_parent_action(action)
                print(f"Padre identificado: {parent_id}")
                
                # Extraer y procesar la acción
                action_data = self.extract_action_properties(action)
                processed_action = self.process_single_action(action_data)
                
                # Almacenar la acción procesada
                if action_id:
                    self.processed_actions[action_id] = processed_action
                    print(f"Acción {action_id} almacenada en processed_actions")
                    
                    # Construir el árbol
                    if parent_id not in self.action_tree:
                        self.action_tree[parent_id] = []
                    
                    self.action_tree[parent_id].append({
                        'id': action_id,
                        'action': processed_action
                    })
                    print(f"Acción {action_id} añadida al árbol bajo el padre {parent_id}")
                
                print(f"Acción {action_id} procesada completamente")
                
        except Exception as e:
            print(f"Error en build_action_tree: {str(e)}")

        print(f"\nÁrbol final construido con {len(self.action_tree)} nodos")
        return self.action_tree

    def process_workflow(self, xml_path: str) -> Dict[str, Any]:
        """
        Procesa el workflow completo de forma segura
        """
        print(f"\nIniciando procesamiento del workflow: {xml_path}")
        root = self.load_workflow(xml_path)
        
        if root is None:
            print("ERROR: No se pudo cargar el workflow")
            return {
                'error': 'No se pudo cargar el workflow',
                'workflow_tree': {},
                'processed_actions': {}
            }
            
        action_tree = self.build_action_tree(root)
        
        result = {
            'workflow_tree': action_tree,
            'processed_actions': self.processed_actions
        }
        
        print(f"\nProcesamiento completado:")
        print(f"- Acciones en el árbol: {len(action_tree)}")
        print(f"- Acciones procesadas: {len(self.processed_actions)}")
        
        return result

    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Guarda los resultados en un archivo JSON de forma segura
        """
        try:
            print(f"\nGuardando resultados en: {output_path}")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print("Resultados guardados correctamente")
            
            # Mostrar resumen de lo guardado
            print("\nResumen de resultados guardados:")
            print(f"- Tamaño del archivo: {os.path.getsize(output_path)} bytes")
            print(f"- Acciones en workflow_tree: {len(results['workflow_tree'])}")
            print(f"- Acciones procesadas: {len(results['processed_actions'])}")
            
        except Exception as e:
            print(f"Error al guardar resultados: {str(e)}")

def main():
    try:
        # Obtener la ruta absoluta al directorio del script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Configurar rutas
        MODEL_PATH = os.path.join(script_dir, "mistral-7b-v0.1.Q4_K_M.gguf")
        INPUT_XML = os.path.join(script_dir, "workflow.xml")
        OUTPUT_JSON = os.path.join(script_dir, "resultado_workflow.json")

        print(f"Configuración inicial:")
        print(f"- Directorio del script: {script_dir}")
        print(f"- Ruta del modelo: {MODEL_PATH}")
        print(f"- Archivo de entrada: {INPUT_XML}")
        print(f"- Archivo de salida: {OUTPUT_JSON}")
        
        # Verificar existencia de archivos
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No se encuentra el modelo en: {MODEL_PATH}")
        if not os.path.exists(INPUT_XML):
            raise FileNotFoundError(f"No se encuentra el archivo XML en: {INPUT_XML}")
        
        # Procesar el workflow
        processor = NintexToPowerAutomate(MODEL_PATH)
        results = processor.process_workflow(INPUT_XML)
        processor.save_results(results, OUTPUT_JSON)
        
    except Exception as e:
        print(f"\nError durante la ejecución: {str(e)}")

if __name__ == "__main__":
    main()
