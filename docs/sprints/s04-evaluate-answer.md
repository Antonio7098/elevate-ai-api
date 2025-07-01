# Sprint 04: Answer Evaluation Endpoint

**Signed off** Antonio
**Date Range:** [Start Date] - [End Date]
**Primary Focus:** Core API - Answer Evaluation Endpoint Implementation
**Overview:** Implement the POST /api/ai/evaluate-answer endpoint that evaluates a user's answer to a question using the internal AI service, returning marks and feedback.

---

## I. Planned Tasks & To-Do List (Derived from Gemini's Prompt)

*Instructions for Antonio: Review the prompt/instructions provided by Gemini for the current development task. Break down each distinct step or deliverable into a checkable to-do item below. Be specific.*

- [ ] **Task 1:** Create the EvaluateAnswerDto schema
    - [ ] *Sub-task 1.1:* Define the DTO structure with required 'questionId' (number) and 'userAnswer' (string)
    - [ ] *Sub-task 1.2:* Add validation for the DTO fields (questionId must be positive integer, userAnswer cannot be empty)
- [ ] **Task 2:** Implement the answer evaluation endpoint handler
    - [ ] *Sub-task 2.1:* Create the POST /api/ai/evaluate-answer route
    - [ ] *Sub-task 2.2:* Add authentication and authorization checks (JWT)
    - [ ] *Sub-task 2.3:* Validate that the questionId exists and user has access
    - [ ] *Sub-task 2.4:* Fetch question, marking criteria, marks available, and context
    - [ ] *Sub-task 2.5:* Construct payload for the AI service (question, expected answer, user answer, type, options, marks, criteria, context)
    - [ ] *Sub-task 2.6:* Call the internal AI service and handle the response
    - [ ] *Sub-task 2.7:* Calculate marks achieved (score × marks available, rounded)
    - [ ] *Sub-task 2.8:* Respond with only the required fields (correctedAnswer, marksAvailable, marksAchieved)
- [ ] **Task 3:** Implement the internal AI service integration
    - [ ] *Sub-task 3.1:* Create the request payload for the AI service with all required fields
    - [ ] *Sub-task 3.2:* Handle the AI service response parsing (score, corrected answer)
    - [ ] *Sub-task 3.3:* Implement error handling for AI service failures (503 Service Unavailable)
- [ ] **Task 4:** Add comprehensive error handling
    - [ ] *Sub-task 4.1:* Handle 400 Bad Request for validation errors
    - [ ] *Sub-task 4.2:* Handle 401 Unauthorized for authentication issues
    - [ ] *Sub-task 4.3:* Handle 404 Not Found when question doesn't exist or user lacks access
    - [ ] *Sub-task 4.4:* Handle 503 Service Unavailable when AI service is down
    - [ ] *Sub-task 4.5:* Handle 500 Internal Server Error for unexpected errors
- [ ] **Task 5:** Create unit tests for the endpoint
    - [ ] *Sub-task 5.1:* Test successful answer evaluation
    - [ ] *Sub-task 5.2:* Test various error scenarios (invalid question, auth failures, missing fields, AI service down)
    - [ ] *Sub-task 5.3:* Test with different question types and marking criteria
- [ ] **Task 6:** Update API documentation and examples
    - [ ] *Sub-task 6.1:* Add endpoint documentation to the API docs
    - [ ] *Sub-task 6.2:* Create example request/response payloads
    - [ ] *Sub-task 6.3:* Document error responses and usage instructions

---

## II. Agent's Implementation Summary & Notes

*Instructions for AI Agent (Cascade): For each planned task you complete from Section I, please provide a summary below. If multiple tasks are done in one go, you can summarize them together but reference the task numbers.*

**Regarding Task 1: EvaluateAnswerDto schema**
* **Summary of Implementation:**
    * [To be filled in after implementation]
* **Key Files Modified/Created:**
    * [To be filled in]
* **Notes/Challenges Encountered (if any):**
    * [To be filled in]

**Regarding Task 2: Endpoint Handler**
* **Summary of Implementation:**
    * [To be filled in after implementation]
* **Key Files Modified/Created:**
    * [To be filled in]
* **Notes/Challenges Encountered (if any):**
    * [To be filled in]

**Regarding Task 3: Internal AI Service Integration**
* **Summary of Implementation:**
    * [To be filled in after implementation]
* **Key Files Modified/Created:**
    * [To be filled in]
* **Notes/Challenges Encountered (if any):**
    * [To be filled in]

**Regarding Task 4: Error Handling**
* **Summary of Implementation:**
    * [To be filled in after implementation]
* **Key Files Modified/Created:**
    * [To be filled in]
* **Notes/Challenges Encountered (if any):**
    * [To be filled in]

**Regarding Task 5: Unit Tests**
* **Summary of Implementation:**
    * [To be filled in after implementation]
* **Key Files Modified/Created:**
    * [To be filled in]
* **Notes/Challenges Encountered (if any):**
    * [To be filled in]

**Regarding Task 6: Documentation and Examples**
* **Summary of Implementation:**
    * [To be filled in after implementation]
* **Key Files Modified/Created:**
    * [To be filled in]
* **Notes/Challenges Encountered (if any):**
    * [To be filled in]

---

## III. Overall Sprint Summary & Review (To be filled out by Antonio after work is done)

**1. Key Accomplishments this Sprint:**
    * [List what was successfully completed and tested]
    * [Highlight major breakthroughs or features implemented]

**2. Deviations from Original Plan/Prompt (if any):**
    * [Describe any tasks that were not completed, or were changed from the initial plan. Explain why.]
    * [Note any features added or removed during the sprint.]

**3. New Issues, Bugs, or Challenges Encountered:**
    * [List any new bugs found, unexpected technical hurdles, or unresolved issues.]

**4. Key Learnings & Decisions Made:**
    * [What did you learn during this sprint? Any important architectural or design decisions made?]

**5. Blockers (if any):**
    * [Is anything preventing progress on the next steps?]

**6. Next Steps Considered / Plan for Next Sprint:**
    * [Briefly outline what seems logical to tackle next based on this sprint's outcome.]

**Sprint Status:** [e.g., Fully Completed, Partially Completed - X tasks remaining, Completed with modifications, Blocked] 