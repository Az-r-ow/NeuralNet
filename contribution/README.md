# Contributions

# Branch Names

The branch name should be a one or two word max (separated with a -) name of the feature.

## Commits convention :

For the main part following whatever's written [here](https://www.conventionalcommits.org/en/v1.0.0/) should be enough.

To bump versions, commits messages should start with :

- **Major** (eg: v1.0.0 -> v2.0.0): `major:`
- **Minor** (eg: v1.0.0 -> v1.1.0): `minor:`
- **Patch** (eg: v1.0.0 -> v1.0.1): `fix:`

### Non Functional Commit :

Non functional commits are when a feature is still under development but you had to create a commit. For example, if you're working on feature A at home on your PC, and all of a sudden you have to save you're changes so you can resume working at a Starbucks on your laptop. You create a commit with the tag : nfc(A) - A should be replaced with the name of the feature (which is the name of the branch).

## Code Style :

I used [google](https://google.github.io/styleguide/cppguide.html)'s style for formatting the code, however I don't apply all google's C++ style guides in the project.

When contributing make sure you format your code with `clang-format`. If you're on VS Code setup the `clang-format` extension and enable `formatOnSave`.
