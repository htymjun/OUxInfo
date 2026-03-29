# Contributing to OUxInfo
Thank you for your interest in contributing to OUxInfo.

We welcome contributions of all kinds, including bug fixes, new features, documentation improvements, and suggestions.

---
## Workflow
Please follow the standard GitHub workflow:
1. **Fork the repository**
2. **Create a new branch**
   ~~~bash
   $ git checkout -b feature/your-feature-name
   ~~~
3. Make your changes
4. Run tests before committing
   ~~~bash
   $ cd /tests
   $ pytest
   ~~~
   All tests must pass.
5. Commit your changes
   ~~~bash
   $ git commit -m "Add: short description of your change."
   ~~~
6. Push to your fork
   ~~~bash
   $ git push origin feature/your-feature-name
   ~~~
7. Open a Pull Request (PR)
   * Clearly describe what you changed and why
   * Link related issues if applicable

## Development Guidelines
### Code Style
* Use 2 spaces for indentation (project-specific convention)
* Do not mix tabs and spaces
* Keep formatting consistent across the codebase
* Use meaningful variable names
* Keep functions small and focused
### Documentation
* Add docstrings to all public functions
* Update README if behavior changes
* Provide usage examples when adding features

## Testing
We use `pytest` for testing.
* Add tests for any new functionality
* Ensure all existing tests pass
* Place tests in the `tests/` directory
Example:
~~~bash
$ pytest
~~~

## Reporting Issues
If you find a bug or have a feature request:
* Open an issue on GitHub
* Provide a clear description
* Include minimal reproducible examples if possible

## Code of Conduct
Please be respectful and constructive in all interactions.

## Questions
If you have any questions, feel free to open an issue or start a discussion.

Thank you for contributing.

