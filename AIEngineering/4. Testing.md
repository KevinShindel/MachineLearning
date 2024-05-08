## Create tests with AI Copilot

To meet goals:
- Use a Code Coverage Tool
- Inspect and Refine the Tests


### Creating Unit Tests
- Task : You must write unit tests to cover code that implements a reverse Polish notation (RPN) calculator.
```java
public static double calculate(String formula) {
        String[] tokens = formula.split(" ");
        Stack<Double> stack = new Stack<>();

        for(String token : tokens) {
            switch(token) {
                case "+":
                    stack.push(stack.pop() + stack.pop());
                    break;
                case "-":
                    stack.push(-stack.pop() + stack.pop());
                    break;
                case "*":
                    stack.push(stack.pop() * stack.pop());
                    break;
                case "/":
                    double divisor = stack.pop();
                    stack.push(stack.pop() / divisor);
                    break;
                default:
                    stack.push(Double.valueOf(token));
            }
        }

        return stack.pop();
    } 
```

- Prompt: Create a set of test cases to ensure the code below is thoroughly tested, considering that the code implements reverse Polish notation (RPN). Then, generate unit tests for these test cases in JUnit5.

```java
  public static double calculate(String formula) {
        String[] tokens = formula.split(" ");
        Stack<Double> stack = new Stack<>();

        for(String token : tokens) {
            switch(token) {
                case"+":
                    stack.push(stack.pop() + stack.pop());
                    break;
                case "-":
                    stack.push(-stack.pop() + stack.pop());
                    break;
                case "*":
                    stack.push(stack.pop() * stack.pop());
                    break;
                case "/":
                    double divisor = stack.pop();
                    stack.push(stack.pop() / divisor);
                    break;
                default:
                    stack.push(Double.valueOf(token));
            }
        }

        return stack.pop();
    }
```

- Output:
```text
Test Cases:

1. Test with a simple addition operation.
2. Test with a simple subtraction operation.
3. Test with a simple multiplication operation.
4. Test with a simple division operation.
5. Test with a combination of multiple operations.
6. Test with a formula containing decimal numbers.
7. Test with a formula containing negative numbers.
8. Test with an empty formula.
9. Test with an invalid formula (e.g., missing an operand or operator).
```

```java
JUnit5 Unit Tests:

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

class RPNCalculatorTest {

    @Test
    void testSimpleAddition() {
        assertEquals(5, RPNCalculator.calculate("2 3 +"), 0.001);
    }

    @Test
    void testSimpleSubtraction() {
        assertEquals(1, RPNCalculator.calculate("4 3 -"), 0.001);
    }

    @Test
    void testSimpleMultiplication() {
        assertEquals(6, RPNCalculator.calculate("2 3 *"), 0.001);
    }

    @Test
    void testSimpleDivision() {
        assertEquals(2, RPNCalculator.calculate("4 2 /"), 0.001);
    }

    @Test
    void testMultipleOperations() {
        assertEquals(14, RPNCalculator.calculate("5 3 + 2 * 4 -"), 0.001);
    }

    @Test
    void testDecimalNumbers() {
        assertEquals(7.5, RPNCalculator.calculate("2.5 3 + 2 *"), 0.001);
    }

    @Test
    void testNegativeNumbers() {
        assertEquals(-1, RPNCalculator.calculate("2 -3 +"), 0.001);
    }

    @Test
    void testEmptyFormula() {
        assertThrows(IllegalArgumentException.class, () -> RPNCalculator.calculate(""));
    }

    @Test
    void testInvalidFormula() {
        assertThrows(IllegalArgumentException.class, () -> RPNCalculator.calculate("2 3 + *"));
    }
}
```

### When using AI conversational tools to generate unit tests, keep in mind the following:
- Include a request to create test cases and unit tests based on these cases in your prompt. This allows you to validate the test cases and decide which are required based on the application's business logic. By analyzing test cases, developers can also understand whether the code implements all the necessary functionality.
- If you're using ChatGPT, set the minimum temperature to 0 (precise) because it makes responses more focused and deterministic, thus allowing you to create more test cases and tests.

### When using AI inline tools to generate unit tests, keep in mind the following:
+ Combining Approaches: It is best to combine different approaches to solve the problem. The GWT method works best most of the time.
+ Provide Context: Without knowledge of what is being tested, GitHub Copilot will produce tests for the given code, even though the goal of testing may be unclear. Sometimes, this can create more confusion than clarity. To provide more context, try to:
  - Keep open some classes that are related to the one being tested
  - Provide meaningful comments in the class being tested and the test class
+ Code Logic: GitHub Copilot tries to comprehend the logic of your code. If it's correct, this tool will be helpful to you. Otherwise, its suggestions will result in useless tests.
+ Business Logic: GitHub Copilot does not understand business logic. When code completely covers business requirements, it will generate the appropriate tests. Otherwise, the tests it generates may validate incorrect principles.
+ Limited Understanding: GitHub Copilot's capabilities are limited to the code segment specified. If the method under test calls other methods in different files, it may resort to guessing.

### Test Maintenance
- Task: Suppose you have source code that has been covered with tests, but the functionality was extended, and now you need to update your unit tests based on the new functionality.
- Prompt: 
    - I have updated my function code to include new functionality. Please update the test cases and unit tests to cover these changes. Ensure that the updated test cases and unit tests follow the provided code style and conventions. The updated function code and previous tests are provided below. Explain what was updated and why.
```java
Function code:

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Main {

    public static String getValueInBracesAndReplace(String stringWithBraces, String newValue, String valueToReplace) {
        if (stringWithBraces == null) {
            return "";
        }

        if (valueToReplace != null && newValue != null) {
            return stringWithBraces.replaceAll("\\{"> + valueToReplace + "\\}", newValue);
        }

        if (newValue == null) {
            return stringWithBraces;
        }

        return stringWithBraces.replaceAll("\\{(.+?)\\}", newValue);
    }

    // Overloaded method for optional newValue and valueToReplace
    public static String getValueInBracesAndReplace(String stringWithBraces) {
        return getValueInBracesAndReplace(stringWithBraces, null, null);
    }

    // Overloaded method for optional valueToReplace
    public static String getValueInBracesAndReplace(String stringWithBraces, String newValue) {
        return getValueInBracesAndReplace(stringWithBraces, newValue, null);
    }

}
```

```java
Previous tests:

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MainTest {

    @Test
    public void shouldReturnInitialStringWithTextInBracesIfNewValueIsUndefined() {
        String stringWithBraces = "{formattedPrice} per seat";
        String newValue = null;
        String resultFields = Main.getValueInBracesAndReplace(stringWithBraces, newValue);
        assertEquals(stringWithBraces, resultFields);
    }

    @Test
    public void shouldReturnEmptyStringIfStringWithTextInBracesIsUndefined() {
        String stringWithBraces = null;
        String newValue = "7.99";
        String newString = "";
        String resultFields = Main.getValueInBracesAndReplace(stringWithBraces, newValue);
        assertEquals(newString, resultFields);
    }

    @Test
    public void shouldReturnStringWithNewValueInsteadOfTextInBraces() {
        String stringWithBraces = "{formattedPrice} per seat";
        String newValue = "7.99";
        String newString = "7.99 per seat";
        String resultFields = Main.getValueInBracesAndReplace(stringWithBraces, newValue);
        assertEquals(newString, resultFields);
    }

}
```

### Output:

I have updated the test cases and unit tests to cover the changes in the function. The updated test cases now include tests for the new functionality of replacing a specific value in the braces and also for the overloaded methods. The updated test cases and unit tests follow the provided code style and conventions.

Updated test cases:

```java
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MainTest {

    @Test
    public void shouldReturnInitialStringWithTextInBracesIfNewValueIsUndefined() {
        String stringWithBraces = "{formattedPrice} per seat";
        String newValue = null;
        String resultFields = Main.getValueInBracesAndReplace(stringWithBraces, newValue);
        assertEquals(stringWithBraces, resultFields);
    }

    @Test
    public void shouldReturnEmptyStringIfStringWithTextInBracesIsUndefined() {
        String stringWithBraces = null;
        String newValue = "7.99";
        String newString = "";
        String resultFields = Main.getValueInBracesAndReplace(stringWithBraces, newValue);
        assertEquals(newString, resultFields);
    }

    @Test
    public void shouldReturnStringWithNewValueInsteadOfTextInBraces() {
        String stringWithBraces = "{formattedPrice} per seat";
        String newValue = "7.99";
        String newString = "7.99 per seat";
        String resultFields = Main.getValueInBracesAndReplace(stringWithBraces, newValue);
        assertEquals(newString, resultFields);
    }

    @Test
    public void shouldReturnStringWithNewValueInsteadOfSpecificValueInBraces() {
        String stringWithBraces = "{formattedPrice} per seat";
        String newValue = "7.99";
        String valueToReplace = "formattedPrice";
        String newString = "7.99 per seat";
        String resultFields = Main.getValueInBracesAndReplace(stringWithBraces, newValue, valueToReplace);
        assertEquals(newString, resultFields);
    }

    @Test
    public void shouldReturnInitialStringWithTextInBracesForOverloadedMethod() {
        String stringWithBraces = "{formattedPrice} per seat";
        String resultFields = Main.getValueInBracesAndReplace(stringWithBraces);
        assertEquals(stringWithBraces, resultFields);
    }

    @Test
    public void shouldReturnStringWithNewValueInsteadOfTextInBracesForOverloadedMethod() {
        String stringWithBraces = "{formattedPrice} per seat";
        String newValue = "7.99";
        String newString = "7.99 per seat";
        String resultFields = Main.getValueInBracesAndReplace(stringWithBraces, newValue);
        assertEquals(newString, resultFields);
    }

}
```
The updates include:

1. Added a test case shouldReturnStringWithNewValueInsteadOfSpecificValueInBraces to test the new functionality of replacing a specific value in the braces.
2. Added test cases shouldReturnInitialStringWithTextInBracesForOverloadedMethod and shouldReturnStringWithNewValueInsteadOfTextInBracesForOverloadedMethod to test the overloaded methods of the `getValueInBracesAndReplace` function.

These updates ensure that the test cases cover all the possible scenarios and functionalities provided by the updated function code. The test cases now test for:

1. Returning the initial string with text in braces if the new value is undefined.
2. Returning an empty string if the string with text in braces is undefined.
3. Returning a string with the new value instead of the text in braces.
4. Returning a string with the new value instead of a specific value in braces.
5. Testing the overloaded method without providing a new value and value to replace.
6. Testing the overloaded method with only providing a new value.

These test cases ensure the updated function code works as expected and covers all the possible scenarios and edge cases.


### Test data generation
Here are some best practices you should follow when using AI to generate test data:
- **Provide examples** - of the intended test data type or structure whenever available. This can help AI better understand your requirements and create test data that meets your needs.
- **Ask for Clarification** - If you are unsure about any aspect of the test data generation process or need more information on how conversational AI tools can help, don't hesitate to ask questions. These tools are designed to provide assistance and guidance, so using these capabilities will ensure you get the most out of the collaboration.

### Generating Test Data in CSV Format
- Task: 
```java
// The model for an array of films is shown in the code sample below. For further unit testing, you need dummy data in .csv format

public class Film {  

    private String filmId;  

    private String filmName;  

    private String description;  

    private String releaseDate;  

    private String distributor;  

    private String cast;  

    private int minutesLength;  

}
```

- Prompt: Generate test data in CSV format to create an array of five objects in the following model:

```java
public class Film {
...  
}
```

- Output:
```csv
filmId,filmName,description,releaseDate,distributor,cast,minutesLength 

1,Film1,Description1,01-01-2020,Distributor1,Cast1,120 

2,Film2,Description2,02-01-2020,Distributor2,Cast2,130 

3,Film3,Description3,03-01-2020,Distributor3,Cast3,140 

4,Film4,Description4,04-01-2020,Distributor4,Cast4,150 

5,Film5,Description5,05-01-2020,Distributor5,Cast5,160
```

### Generating Test Data in Source Code

- Task: 
```java
The code sample below illustrates the model for an array of films. For further unit testing, you need a code snippet that initializes the data array directly in the source code.

public class Film {  

    private String filmId;  

    private String filmName;  

    private String description; 
  
    private String releaseDate;  

    private String distributor;  

    private String cast;  

    private int minutesLength;
 
 
    public Film(String filmId, String filmName, String description, String releaseDate, String distributor, String cast, int minutesLength) {  

        this.filmId = filmId;  

        this.filmName = filmName;  

        this.description = description;  

        this.releaseDate = releaseDate;  

        this.distributor = distributor;  

        this.cast = cast;  

        this.minutesLength = minutesLength;  

    }  

}
```

- Prompt: Generate a Java code snippet to create an array of five Film objects with test data, as shown in the following code sample:
```java
public class Film {
...  
}
```

- Output: 
```java
public class Main { 

public static void main(String[] args) { 

    Film[] films = new Film[5]; 

    films[0] = new Film("F001","Test Film 1", "A test film with random data", "2022-01-01", "Distributor 1", "Actor 1, Actor 2, Actor 3", 120); 

    films[1] = new Film("F002", "Test Film 2", "Another test film with random data", "2022-02-01", "Distributor 2", "Actor 4, Actor 5, Actor 6", 100); 

    films[2] = new Film("F003", "Test Film 3", "A third test film with random data", "2022-03-01", "Distributor 3", "Actor 7, Actor 8, Actor 9", 150); 

    films[3] = new Film("F004", "Test Film 4", "A fourth test film with random data", "2022-04-01", "Distributor 4", "Actor 10, Actor 11, Actor 12", 90); 

    films[4] = new Film("F005", "Test Film 5", "A fifth test film with random data", "2022-05-01", "Distributor 5", "Actor 13, Actor 14, Actor 15", 110); 

} 

}
This Java code creates an array of 5 Film objects with the specified properties. The Film class is defined with the given fields and constructor, and the Main class contains the main method that creates the Film objects and displays their information.
```

### When using AI conversational tools to generate new test data, keep in mind the following:
1. **Use Domain-Specific Language**: terminology in your prompt to leverage the model's knowledge of programming languages, libraries, and frameworks. This helps the model understand the context better and generate more accurate test data.
2. Leverage the Model's Knowledge: GPT models have a vast knowledge of various domains. You can ask the model to generate test data based on real-world examples or popular datasets. For example, "Generate five rows of test data for movies similar to popular Hollywood films."
3. Control Randomness: f you want the model to generate more diverse or random test data, you can instruct it explicitly to do so. For example, "Generate five rows of test data for movies with diverse genres, casts, and release dates."
4. Use Constraints: You can provide constraints or conditions that the generated test data must meet. For example, "Generate five rows of test data for movies with a release date between 2010 and 2020 and a length between 90 and 180 minutes."

### Updating Existing Test Data
- Task: Your model for the film was extended to a new version. The new version includes the following fields: genre, picture, trailerUrl, and certificates. Now, you need to update your test data according to the new model.
- Prompt:I have updated my data model for films to a new version. Please update my existing test data according to the new data model. The updated data model and existing test data are provided below. Explain what was updated and why.
```java
// Updated model description:
public class Certificate { 
    private String src; 
    private String name; 
    public Certificate(String src, String name) { 
        this.src = src; 
        this.name = name; 
    } 
    // Getters and Setters 
} 
public class Film { 
    private String filmId; 
    private String filmName; 
    private String description; 
    private String releaseDate; 
    private String distributor; 
    private String genre; 
    private String cast; 
    private int minutesLength; 
    private String picture; 
    private String trailerUrl; 
    private Certificate[] certificates; 
    public Film(String filmId, String filmName, String description, String releaseDate, String distributor, String genre, String cast, int minutesLength, String picture, String trailerUrl, Certificate[] certificates) { 
        this.filmId = filmId; 
        this.filmName = filmName; 
        this.description = description; 
        this.releaseDate = releaseDate; 
        this.distributor = distributor; 
        this.genre = genre; 
        this.cast = cast; 
        this.minutesLength = minutesLength; 
        this.picture = picture; 
        this.trailerUrl = trailerUrl; 
        this.certificates = certificates; 
    } 
    // Getters and Setters 
}
Existing test data: 
public class Main { 
    public static void main(String[] args) { 
        Film[] films = new Film[5]; 
        films[0] = new Film("1", "Film1", "Description1", "01-01-2020", "Distributor1", "Cast1", 120); 
        films[1] = new Film("2", "Film2", "Description2", "02-01-2020", "Distributor2", "Cast2", 130); 
        films[2] = new Film("3", "Film3", "Description3", "03-01-2020", "Distributor3", "Cast3", 140); 
        films[3] = new Film("4", "Film4", "Description4", "04-01-2020", "Distributor4", "Cast4", 150); 
        films[4] = new Film("5", "Film5", "Description5", "05-01-2020", "Distributor5", "Cast5", 160); 

} 
} 
```
- Fine-tuning: The output might not be what exactly what you wanted. For example, in the preceding prompt, you considered not one but numerous certificates for each film. Continue the conversation with the tool and ask it to update the code to reflect the new requirements.
  Update this code so that each film has multiple certificates (from one to five certificates per film).
- Output:
```java
Certainly! Here's the updated code with multiple certificates for each film:

public class Main {
    public static void main(String[] args) {

        Film[] films = new Film[5];

        Certificate[] certificates1 = {
            new Certificate("src1_1", "Certificate1_1"),
            new Certificate("src1_2", "Certificate1_2")
        };
        films[0] = new Film("1", "Film1", "Description1", "01-01-2020", "Distributor1", "Genre1", "Cast1", 120, "Picture1", "TrailerUrl1", certificates1);

        Certificate[] certificates2 = {
            new Certificate("src2_1", "Certificate2_1"),
            new Certificate("src2_2", "Certificate2_2"),
            new Certificate("src2_3", "Certificate2_3")
        };
        films[1] = new Film("2", "Film2", "Description2", "02-01-2020", "Distributor2", "Genre2", "Cast2", 130, "Picture2", "TrailerUrl2", certificates2);

        Certificate[] certificates3 = {
            new Certificate("src3_1", "Certificate3_1"),
            new Certificate("src3_2", "Certificate3_2"),
            new Certificate("src3_3", "Certificate3_3"),
            new Certificate("src3_4", "Certificate3_4")
        };
        films[2] = new Film("3", "Film3", "Description3", "03-01-2020", "Distributor3", "Genre3", "Cast3", 140, "Picture3", "TrailerUrl3", certificates3);

        Certificate[] certificates4 = {
            new Certificate("src4_1", "Certificate4_1"),
            new Certificate("src4_2", "Certificate4_2"),
            new Certificate("src4_3", "Certificate4_3"),
            new Certificate("src4_4", "Certificate4_4"),
            new Certificate("src4_5", "Certificate4_5")
        };
        films[3] = new Film("4", "Film4", "Description4", "04-01-2020", "Distributor4", "Genre4", "Cast4", 150, "Picture4", "TrailerUrl4", certificates4);

        Certificate[] certificates5 = {
            new Certificate("src5_1", "Certificate5_1"),
            new Certificate("src5_2", "Certificate5_2"),
            new Certificate("src5_3", "Certificate5_3")
        };
        films[4] = new Film("5", "Film5", "Description5", "05-01-2020", "Distributor5", "Genre5", "Cast5", 160, "Picture5", "TrailerUrl5", certificates5);

    }
}
In this updated code, I added multiple Certificate objects for each film. Film1 has two certificates, Film2 has three certificates, Film3 has four certificates, Film4 has five certificates, and Film5 has three certificates.
```