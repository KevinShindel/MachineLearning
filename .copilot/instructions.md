# Copilot Instructions for ML Project

## Context Files
When working on this project, always reference these documentation files:

### Testing
- **File**: `tests.instructions.md`
- **When to use**: Building, refactoring, or reviewing any test code
- **How**: Read this file first to understand testing standards and patterns

---

## Testing Workflow for Copilot Agent

### When User Asks to Create Tests:
1. ✅ Read `tests.instructions.md` 
2. ✅ Follow the naming conventions from section 3
3. ✅ Use test structure from section 4
4. ✅ Apply patterns from section 11
5. ✅ Validate against section 14 (best practices)

### Example Request Handling:
**User**: "Create tests for data_processing.py"

**Copilot Action**:
- [ ] Read `tests.instructions.md` (sections 3-4)
- [ ] Check file structure (section 2)
- [ ] Apply data processing test pattern (section 4 - "For Data Processing")
- [ ] Ensure naming follows convention (e.g., `test_data_processing.py`)
- [ ] Validate coverage and assertions (section 11)

---

## Testing Standards Summary

### Naming
- Test files: `test_<module>.py`
- Test classes: `Test<Name>`
- Test methods: `test_<behavior>`

### Structure