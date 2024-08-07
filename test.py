import curses
from data.utils import get_european_schedule


def main_menu(stdscr, options):
    # Turn off cursor and initialize colors
    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)

    current_row = 0

    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "Select an option for league ID from the following:")

        # Display the menu options with highlighting
        for idx, row in enumerate(options):
            if idx == current_row:
                stdscr.attron(curses.color_pair(1))
                stdscr.addstr(idx + 2, 0, f"* {row}")  # Move options down by one row
                stdscr.attroff(curses.color_pair(1))
            else:
                stdscr.addstr(idx + 2, 2, row)  # Move options down by one row

        stdscr.refresh()

        key = stdscr.getch()

        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(options) - 1:
            current_row += 1
        elif key == curses.KEY_ENTER or key in [10, 13]:
            selected_option = options[current_row]
            stdscr.addstr(8, 0, f"You selected: {selected_option}")
            stdscr.refresh()

            # Ask for an integer input using curses
            stdscr.addstr(10, 0, "Enter an integer code: ")
            curses.echo()  # Enable echoing of typed characters
            int_code = stdscr.getstr(
                11, 0, 20
            )  # Get user input, max length of 20 characters
            int_code = int(int_code.decode("utf-8"))  # Decode and convert to integer

            stdscr.addstr(0, 0, f"You entered: {int_code}")
            return selected_option, int_code


def main():
    # supported_leagues = [
    #     "ENG-Premier League",
    #     "ESP-La Liga",
    #     "FRA-Ligue 1",
    #     "GER-Bundesliga",
    #     "ITA-Serie A",
    # ]

    # # Initialize curses and show the menu
    # selected_option, int_code = curses.wrapper(main_menu, supported_leagues)
    # print(f"You selected: {selected_option}")

    # # Ask for an integer input
    # print(f"You entered: {int_code}")

    # print("Starting web scraper...")
    print(get_european_schedule(1718))
    return


if __name__ == "__main__":
    main()
