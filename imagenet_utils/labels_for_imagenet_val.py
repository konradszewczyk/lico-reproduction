import os
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm


def parse_xml(xml_file):
    """Parse the XML file to extract filename and name."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    name = root.find('.//object/name').text
    return filename, name


def create_csv_from_xml(folder_path, output_csv):
    """Create a CSV file from XML files in the specified folder."""
    data = []

    # Loop through all XML files in the folder
    for file in tqdm(os.listdir(folder_path)):
        if file.endswith('.xml'):
            xml_file = os.path.join(folder_path, file)
            filename, name = parse_xml(xml_file)
            data.append([filename, name])

    # Create a DataFrame and save it to a CSV file
    df = pd.DataFrame(data, columns=['Filename', 'Class'])
    df.to_csv(output_csv, index=False)

folder_path = 'path/to/your/xml/files'
output_csv = 'output.csv'

create_csv_from_xml(folder_path, output_csv)
