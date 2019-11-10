import re
import glob

# Find all source files
file_list = list(glob.iglob('./fbpic/**/*.py', recursive=True)) + list(glob.iglob('./tests/**/*.py', recursive=True))

# Loop over source files
for filename in file_list:

    # Open file
    print(filename)
    with open(filename) as f:
        text = f.read()

    # Remove previous conversion from cuda array to cupy array
    text = re.sub(r'cupy\.asarray\((.*?)\)', r'\1', text)

    # Replace copies to host by the cupy equivalent (`get`)
    text = re.sub('\.copy_to_host\(\)', '.get()', text)
    text = re.sub(r'\.copy_to_host\((\s*)(.*)(\s*)\)', r'.get(\1out=\2\3)', text)

    # Replace copies to device by the cupy equivalent (`set`)
    text = re.sub(r'\.copy_to_device\((\s*)(.*)(\s*)\)', r'.set(\1\2\3)', text)
    text = re.sub(r'cuda\.to_device\((.*?)\, to\=(.*?)\s*\)', r'\2.set(\1)', text)

    # Replace copies to device
    text = re.sub(r'cuda\.to_device\((.*?)\)', r'cupy.asarray(\1)', text)

    # Replace array allocation
    text = re.sub(r'cuda\.device_array_like', r'cupy.asarray', text)
    text = re.sub(r'cuda\.device_array', r'cupy.empty', text)

    # Remove freeing memory
    text = re.sub(r'.*cupy_mempool.*\n', r'', text)

    # Replace getitem
    text = re.sub(r'(\S+)\.getitem\((.*)\)', r'int(\1[\2])', text)

    # Fix import statements
    if ('cupy.' in text) and re.search('import.*cupy', text) is None:
        if re.search('if cupy_installed:', text) is not None:
            text = re.sub(r'if cupy_installed:',
                   r'if cupy_installed:\n    import cupy', text)
        elif re.search('\nif cuda_installed:', text) is not None:
            text = re.sub(r'\nif cuda_installed:',
                   r'\nif cupy_installed:\n    import cupy\nif cuda_installed:',
                          text, count=1)
            text = re.sub(r'import cuda_installed', 'import cupy_installed, cuda_installed', text)
        elif re.search('from fbpic\.utils\.cuda import (.*)cuda([^_])', text) is not None:

            text = re.sub(r'from fbpic\.utils\.cuda import (.*)cuda([^_])',
                   r'from fbpic.utils.cuda import \1cupy, cuda\2', text)
        elif re.search('from numba import cuda', text) is not None:
            text = re.sub(r'from numba import cuda',
                          r'from numba import cuda\nfrom fbpic.utils.cuda import cupy_installed\nif cupy_installed:\n    from fbpic.utils.cuda import cupy', text)

    if not 'cuda.' in text:
        text = re.sub(r'from numba import cuda\n', '', text)
        text = re.sub(r'import(.*)\, cuda([^_])',
                      r'import\1\2', text)
        text = re.sub(r'import(.*) cuda\, ',
                      r'import\1 ', text)

    # Hand-taylored
    if filename.endswith('cuda_sorting.py'):
        text = re.sub(r'.*import cupy\n', '', text)
    if filename.endswith('compton.py'):
        text = re.sub(r'from numba import cuda\n', '', text)
    if filename.endswith('checkpoint_restart.py'):
        text = re.sub(r'import(.*), cuda_installed', r'import\1', text)
        text = re.sub(r'if cuda_installed:\n.*\n', r'', text)

    # Close file
    with open(filename, 'w') as f:
        f.write(text)
