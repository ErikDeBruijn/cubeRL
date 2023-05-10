COLORS = { "reset": "\033[0m", "green": "\033[30;42m"}

def print_colored(cube_state_str):
    def colored_char(char):
        color_codes = {
            'W': "\033[30;107m",         # White background, black text
            'Y': "\033[30;103m",        # Yellow background, black text
            'B': "\033[94;48;5;18m",    # Even darker blue background, black text
            'G': "\033[30;42m",         # Green background, black text
            'R': "\033[30;41m",         # Red background, black text
            'O': "\033[30;48;5;208m"    # Orange-like background, white text
        }

        if char in color_codes:
            return f"{color_codes[char]}{char}{COLORS['reset']}"
        else:
            return char

    colored_cube_state_lines = []
    for line in str(cube_state_str).splitlines():
        colored_line = ''.join([colored_char(c) for c in line])
        colored_cube_state_lines.append(colored_line)

    print('\n'.join(colored_cube_state_lines))