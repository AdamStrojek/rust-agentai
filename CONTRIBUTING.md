# Contributing to AgentAI

Thank you for considering contributing to AgentAI! We welcome contributions from the community to help improve this project.
Whether you want to add a new feature, fix bugs, or improve the documentation, your help is appreciated.

## How to Contribute

### 1. Fork the Repository

Start by forking the repository to your own GitHub account. You can do this by clicking the "Fork" button in the top
right corner of the project’s GitHub page.

### 2. Clone Your Fork

Clone your fork to your local machine using the following command:

```bash
git clone https://github.com/<your-username>/rust-agentai.git
```

### 3. Create a New Branch

Before making any changes, create a new branch to work on. It is best practice to give the branch a descriptive name
that reflects the changes you’re making, like feature-add-agent-type or bugfix-fix-crash.

git checkout -b <branch-name>

### 4. Make Changes

Make your changes in the codebase. Be sure to follow the existing coding style and conventions.
Ensure that your changes do not break existing functionality.

- Add new features or fix bugs.
- If you add new features, consider adding tests.
- If you’re fixing a bug, ensure that you have tested your fix.
- Keep your changes focused and avoid unrelated modifications.

### 5. Commit Your Changes

Once you’ve made your changes, commit them with a clear, concise commit message explaining what you’ve done.
Use the following format for your commit messages:

```
<type>(<scope>): <message>

<optional body of the commit message>
```

Where:
- `<type>` could be feat, fix, docs, style, refactor, test, etc.
- `<scope>` (optional) is the area of the code that the change relates to (e.g., agent, docs, etc.).

Example:

```feat(agent): add support for JSON Schema```

### 6. Push Your Changes

Push your changes to your fork on GitHub:

git push origin <branch-name>

### 7. Create a Pull Request

Go to the original repository (AgentAI) and click on the “Pull Requests” tab. Then click on the “New Pull Request” button.

Select the branch you just pushed from your fork, and compare it to the main branch of the original repository.
Write a detailed description of the changes you’ve made and why they are needed.

Once everything looks good, click “Create Pull Request”.

### 8. Code Review

Once you’ve submitted your pull request, the maintainers will review your changes. You may be asked to make changes based on feedback.
Please be responsive to the comments and work on any requested changes.

### 9. Merge

Once your pull request is approved, it will be merged into the main codebase.

## Code of Conduct

We ask that all contributors adhere to the following code of conduct to ensure a positive and respectful environment for everyone:
- Be respectful and kind to others.
- Communicate openly and constructively.
- vide thoughtful and thorough feedback.
- open to receiving feedback.

## Testing

Before submitting your pull request, make sure that the tests are passing and that your changes don’t introduce any regressions.

To run the tests locally, use:

```bash
cargo test
```

## Documentation

If your change affects the public API or adds new features, please ensure that the documentation is updated accordingly.
You can find the documentation for the project here.

## Thank You!

Thank you for contributing to AgentAI! We appreciate your time and effort in making this project better.

If you have any questions or need help, feel free to open an issue or contact the maintainers.
