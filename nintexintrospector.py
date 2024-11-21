import xml.etree.ElementTree as ET
from llama_cpp import Llama
import json
from typing import Dict, List, Any
import numpy as np
from pathlib import Path


class NintexToPowerAutomate:
    def __init__(self, model_path: str):
        """
        Inicializa el procesador con un modelo LLM local
        """
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=8,
            embedding=True
        )
        self.action_tree = {}
        self.processed_actions = {}

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
        Procesa una única acción usando el modelo LLM
        """
        prompt = f"""
        Analiza esta acción de Nintex Workflow y genera su equivalente en Power Automate.
        Acción: {json.dumps(action_data, indent=2)}
        Genera solo el JSON de la acción equivalente en Power Automate.
        """

        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=512,
            temperature=0.1,
            stop=["}"]
        )

        try:
            power_automate_action = json.loads(response['choices'][0]['text'] + "}")
        except json.JSONDecodeError:
            power_automate_action = {
                "error": "No se pudo generar una traducción válida",
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
    MODEL_PATH = "llama-2-7b.Q4_K_M.gguf"
    INPUT_XML = "workflow_nintex.xml"
    OUTPUT_JSON = "resultado_workflow.json"

    # Procesar el workflow
    processor = NintexToPowerAutomate(MODEL_PATH)
    results = processor.process_workflow(INPUT_XML)
    processor.save_results(results, OUTPUT_JSON)


if __name__ == "__main__":
    main()
