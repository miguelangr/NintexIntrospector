import xml.etree.ElementTree as ET
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from typing import Dict, List, Any
import torch
from pathlib import Path

class NintexToPowerAutomate:
    def __init__(self, model_name: str):
        """
        Inicializa el procesador con un modelo de Hugging Face
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.action_tree = {}
        self.processed_actions = {}

    def generate_response(self, prompt: str) -> str:
        """
        Genera una respuesta usando el modelo
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_return_sequences=1,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
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
        Acción: {json.dumps(action_data, indent=2)}
        Genera solo el JSON de la acción equivalente en Power Automate.
        """

        try:
            response = self.generate_response(prompt)
            # Extraer solo la parte JSON de la respuesta
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > 0:
                json_response = response[json_start:json_end]
                power_automate_action = json.loads(json_response)
            else:
                raise json.JSONDecodeError("No JSON found", response, 0)
                
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
        for action in root.findall('.//NWActionConfig'):
            action_id = action.get('id', '')
            parent_id = self.find_parent_action(action)
            
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

        return self.action_tree

    def process_workflow(self, xml_path: str) -> Dict[str, Any]:
        """
        Procesa el workflow completo y genera la traducción
        """
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
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

def main():
    # Configuración
    MODEL_NAME = "meta-llama/Llama-2-7b"  # o el modelo que prefieras
    INPUT_XML = "ruta/al/workflow.xml"
    OUTPUT_JSON = "resultado_workflow.json"

    # Procesar el workflow
    processor = NintexToPowerAutomate(MODEL_NAME)
    results = processor.process_workflow(INPUT_XML)
    processor.save_results(results, OUTPUT_JSON)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")
