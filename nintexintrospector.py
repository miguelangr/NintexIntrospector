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
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="mistral",
            context_length=2048,
            max_new_tokens=512,
            temperature=0.1
        )
        self.action_tree = {}
        self.processed_actions = {}

    def generate_response(self, prompt: str) -> str:
        """
        Genera una respuesta usando el modelo
        """
        response = self.llm(
            prompt,
            max_new_tokens=512,
            temperature=0.1,
            stop=["}"]  # Para asegurar que el JSON se complete correctamente
        )
        return response.strip()

    def load_workflow(self, xml_path: str) -> ET.Element:
        """
        Carga el archivo XML del workflow
        """
        tree = ET.parse(xml_path)
        return tree.getroot()

    def extract_action_properties(self, action: ET.Element) -> Dict[str, Any]:
        """
        Extrae todas las propiedades de una acción NWActionConfig
        """
        properties = {}
        for prop in action.findall('.//property'):
            prop_name = prop.get('name', '')
            prop_value = prop.text or ''
            properties[prop_name] = prop_value
        
        return {
            'id': action.get('id', ''),
            'name': action.get('name', ''),
            'type': action.get('type', ''),
            'properties': properties
        }

    def process_single_action(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa una única acción usando el modelo
        """
        prompt = f"""
        Analiza esta acción de Nintex Workflow y genera su equivalente en Power Automate.
        Genera solo el JSON de la acción equivalente en Power Automate.
        
        Acción Nintex:
        {json.dumps(action_data, indent=2)}
        
        Respuesta en formato JSON:
        {{
        """

        try:
            response = self.generate_response(prompt)
            # Asegurarse de que tenemos un JSON válido
            json_str = "{" + response.split("{", 1)[-1]
            if not json_str.endswith("}"):
                json_str += "}"
            power_automate_action = json.loads(json_str)
                
        except (json.JSONDecodeError, Exception) as e:
            power_automate_action = {
                "error": f"No se pudo generar una traducción válida: {str(e)}",
                "original_action": action_data
            }

        return power_automate_action

    def find_parent_action(self, action: ET.Element) -> str:
        """
        Encuentra el ID del padre de una acción
        """
        parent = action.getparent()
        while parent is not None:
            if parent.tag == 'NWActionConfig':
                return parent.get('id', '')
            parent = parent.getparent()
        return 'root'

    def build_action_tree(self, root: ET.Element) -> Dict[str, List[Dict[str, Any]]]:
        """
        Construye el árbol de acciones y procesa cada una
        """
        print("Iniciando procesamiento del árbol de acciones...")
        
        for action in root.findall('.//NWActionConfig'):
            action_id = action.get('id', '')
            parent_id = self.find_parent_action(action)
            
            print(f"Procesando acción: {action_id}")
            
            # Extraer y procesar la acción
            action_data = self.extract_action_properties(action)
            processed_action = self.process_single_action(action_data)
            
            # Almacenar la acción procesada
            self.processed_actions[action_id] = processed_action
            
            # Construir el árbol
            if parent_id not in self.action_tree:
                self.action_tree[parent_id] = []
            
            self.action_tree[parent_id].append({
                'id': action_id,
                'action': processed_action
            })
            
            print(f"Acción {action_id} procesada correctamente")

        return self.action_tree

    def process_workflow(self, xml_path: str) -> Dict[str, Any]:
        """
        Procesa el workflow completo y genera la traducción
        """
        print(f"Iniciando procesamiento del workflow: {xml_path}")
        root = self.load_workflow(xml_path)
        action_tree = self.build_action_tree(root)
        
        return {
            'workflow_tree': action_tree,
            'processed_actions': self.processed_actions
        }

    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Guarda los resultados en un archivo JSON
        """
        print(f"Guardando resultados en: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("Resultados guardados correctamente")

def main():
    # Obtener la ruta absoluta al directorio del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Configurar rutas
    MODEL_PATH = os.path.join(script_dir, "mistral-7b-v0.1.Q4_K_M.gguf")
    INPUT_XML = os.path.join(script_dir, "workflow.xml")
    OUTPUT_JSON = os.path.join(script_dir, "resultado_workflow.json")

    print(f"Usando modelo en: {MODEL_PATH}")
    
    # Procesar el workflow
    processor = NintexToPowerAutomate(MODEL_PATH)
    results = processor.process_workflow(INPUT_XML)
    processor.save_results(results, OUTPUT_JSON)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")
