import sys
from PyQt6.QtWidgets import QApplication
from interface.GraspPlannerGUI import GraspPlannerGUI


def main():
    """
    Launch the Grasp Planner GUI application.

    Initializes the PyQt6 application, creates and displays the main GUI window
    for grasp planning experiments, and enters the application event loop.
    """
    app = QApplication(sys.argv)
    window = GraspPlannerGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
