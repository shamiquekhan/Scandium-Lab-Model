import re
import os

def parse_and_build(md_file):
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Look for headings like `path/to/file` followed by a code block
    # Regex explanation:
    # ### .*? `(.*?)`        -> matches the header containing the filepath
    # .*?                    -> matches anything in between (non-greedy)
    # ```[a-z]*\n(.*?)\n```  -> matches the code block
    
    pattern = re.compile(r'### [^\n]*?`([^`]+)`[^\n]*\n.*?\n```[a-z]*\n(.*?)\n```', re.DOTALL)
    
    matches = pattern.findall(content)
    
    for filepath, code in matches:
        # Some filepaths might just be filenames if they are in the root, but the guide seems to give relative paths
        # Let's clean up the filepath
        filepath = filepath.strip()
        print(f"Creating {filepath}...")
        
        # Create directory
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
            
        with open(filepath, 'w', encoding='utf-8') as out_f:
            out_f.write(code)

    # Let's also check for requirements.txt, Dockerfile etc which might be under ## 2.3 `requirements.txt`
    pattern_h2 = re.compile(r'## [^\n]*?`([^`]+)`[^\n]*\n.*?\n```[a-z]*\n(.*?)\n```', re.DOTALL)
    matches_h2 = pattern_h2.findall(content)
    for filepath, code in matches_h2:
        filepath = filepath.strip()
        print(f"Creating {filepath}...")
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as out_f:
            out_f.write(code)

if __name__ == "__main__":
    parse_and_build("scandium_labs_build_guide.md")
