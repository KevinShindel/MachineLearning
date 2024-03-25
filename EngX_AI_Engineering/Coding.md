## Introduction to Coding

```text
As a developer, working with code is a core responsibility, and finding ways to perform tasks more efficiently and effectively is crucial.
 Generative AI can be utilized through both inline and conversational tools to assist with various coding tasks.

The coding process can be broadly divided into two main components: creating new code and modifying existing code.
 Modifying existing code may involve fixing bugs or refactoring to improve the code's quality, maintainability, and readability
```

## Task Research with ChatGPT

To perform a task effectively, the task or problem statement must be understood completely.
 
While researching the task of creating a new feature, keep the following points in mind:
+ Functional aspects—essential task requirements that cannot be overlooked to develop new features properly
+ Components—the fundamental code structure necessary to complete a task

#### Task research with ChatGPT can be broken down into the following steps:
1. Requirements
2. Code Structure
3. Code Implementation
4. Testing
5. Deployment


Task decomposition:
1. Understand the requirements
2. Break down the task into smaller components
3. Creating a hierarchy of components
4. Specifying Dependencies
5. Implement the components
6. Test the components

### Creating a New Function or Method With GitHub Copilot
+ Task: You need to add a feature to display a list of names alphabetically, using a bubble sort method that accepts a list of strings (names) and returns a sorted list.
+ Prompt: Generate a Python method that implements bubble sort. The method should accept a list of strings and return a sorted list.
+ Output: A Python method that implements bubble sort and returns a sorted list of strings.

```python
def bubble_sort(lst):
    n = len(lst)

    for i in range(n):
        for j in range(0, n - i - 1):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
    return lst

names = ["Alice", "Bob", "Charlie", "David", "Eve"]

sorted_names = bubble_sort(names)
print(sorted_names)
```

### Creating a New Class

+ Task: Imagine you've been assigned a task to create a class with three sorting methods for educational purposes—to demonstrate how they work and to compare their effectiveness. Each method should accept a list of strings.
+ Prompt: Generate a Java class called SortingService that contains three approaches to sorting: bubble sort, selection sort, and insertion sort. Ensure this class provides highly readable code.
+ Output: A Java class called SortingService with three sorting methods: bubbleSort, selectionSort, and insertionSort.

```java
import java.util.ArrayList;  
import java.util.List;  

public class SortingService {  
    public static List<String> bubbleSort(List<String> inputList) {  
        int n = inputList.size();  
        String temp;  

        for (int i = 0; i < n - 1; i++) {  
            for (int j = 0; j < n - i - 1; j++) {  
                if (inputList.get(j).compareTo(inputList.get(j + 1)) > 0) {  
                    // Swap elements  
                    temp = inputList.get(j);  
                    inputList.set(j, inputList.get(j + 1));  
                    inputList.set(j + 1, temp);  
                }  
            }  
        }  

        return inputList;  
    }  

    public static List<String> selectionSort(List<String> inputList) {  
        int n = inputList.size();  

        for (int i = 0; i < n - 1; i++) {  
            int minIndex = i;  
            for (int j = i + 1; j < n; j++) {  
                if (inputList.get(j).compareTo(inputList.get(minIndex)) < 0) {  
                    minIndex = j;  
                }  
            }  
            // Swap elements  
            String temp = inputList.get(i);  
            inputList.set(i, inputList.get(minIndex));  
            inputList.set(minIndex, temp);  
        }  

        return inputList;  
    }  

    public static List<String> insertionSort(List<String> inputList) {  
        int n = inputList.size();  

        for (int i = 1; i < n; i++) {  
            String key = inputList.get(i);  
            int j = i - 1;  

            while (j >= 0 && inputList.get(j).compareTo(key) > 0) {  
                inputList.set(j + 1, inputList.get(j));  
                j--;  
            }  

            inputList.set(j + 1, key);  
        }  

        return inputList;  
    }  

    public static void main(String[] args) {  
        List<String> testData = new ArrayList<>();  
        testData.add("banana");  
        testData.add("apple");  
        testData.add("cherry");  
        testData.add("date");  
        testData.add("grape");  

        System.out.println("Original List: " + testData);  
        
        System.out.println("Bubble Sorted List: " + bubbleSort(new ArrayList<>(testData)));  
        System.out.println("Selection Sorted List: " + selectionSort(new ArrayList<>(testData)));  
        System.out.println("Insertion Sorted List: " + insertionSort(new ArrayList<>(testData)));  
    }  
}
```

### Creating a New Module
+ Task: Suppose you are assigned a task to create a REST application to sort and store a list of strings in files.
+ Ptompt: As a Ruby developer, I need assistance creating a new project for a simple REST application to sort and store a list of strings in files. The project should integrate Sinatra for RESTful API development. Each request must be saved in one file, and the name of each file should be a timestamp. Please provide the initial code structure, including the necessary classes and methods, and relevant design patterns.
+ Output: Creating a RESTful API in Ruby can be done using the Sinatra framework. Sinatra is a DSL for quickly creating web applications in Ruby with minimal effort. For storing the sorted list of strings in files, we can use Ruby's built-in File class.

```ruby
# app.rb
require 'sinatra'
require 'json'
require_relative 'string_sorter'

post '/sort' do
  content_type :json
  data = JSON.parse(request.body.read)
  sorter = StringSorter.new(data)
  sorter.sort_and_save
  sorter.sorted_data.to_json
end
```

```ruby
# string_sorter.rb
class StringSorter
  attr_reader :sorted_data

  def initialize(data)
    @data = data
    @sorted_data = []
  end

  def sort_and_save
    @sorted_data = @data.sort
    timestamp = Time.now.to_i
    File.open("#{timestamp}.txt", 'w') do |f|
      f.puts(@sorted_data)
    end
  end
end
```
```text
In this code, we have a POST /sort endpoint that accepts a JSON array of strings, sorts it, saves it to a file, and returns the sorted data.

The StringSorter class is responsible for sorting the data and saving it to a file. The filename is a timestamp, which ensures that each request is saved in a unique file.

To run the application, you would need to install Sinatra (gem install sinatra) and then run ruby app.rb.

Please note that this is a very basic structure and does not include error handling, input validation, or tests, which would be necessary for a production application.
```


### Creating a New Class and Service With GitHub Copilot
+ Method for Latest Exchange Rates : 
```text
This method retrieves the most recent exchange rates for various currencies. The following is the method signature:

public Map<String, Double> getExchangeRates() {
    // Implementation goes here
}
```
+ Method for Historical Exchange Rates: 
```text

This method returns exchange rates for a specific date in the past. It takes a date as a parameter and returns the exchange rates for that date. The following is the method signature:

public Map<String, Double> getExchangeRates(String date) {
    // Implementation goes here
}   
```

### Creating a New Class

The service is created by first defining a new class called OpenExchangeService, annotated with @Service. This class will function as a dedicated service, communicating with openexchangerates.org and retrieving the necessary exchange rate data.

The following prompt can be used as a comment within the code to describe to GitHub Copilot what this service is supposed to do:

```java
    // This service is used to communicate with openexchangerates.org
```


```java
//This service is used to communicate with openexchangerates.org
//It uses the Spring RestTemplate to make HTTP requests
//RestTemplate will be provided from RestTemplateUtil class

import org.springframework.beans.factory.annotation.Autowired;

import java.util.Map;

public class ExchangeRateService {

    @Autowired
    private RestTemplate restTemplate;

    private static final String API_KEY = "YOUR_API";
    private static final String BASE_URL = "https://openexchangerates.org/api";
    
    public Map<String, Double> getExchangeRates() {
        String url = BASE_URL + "/latest.json?app_id=" + API_KEY;
        return restTemplate.getForObject(url, Map.class);
    }
    
    public Map<String, Double> getExchangeRates(String date) {
        String url = BASE_URL + "/historical/" + date + ".json?app_id=" + API_KEY;
        return restTemplate.getForObject(url, Map.class);
    }
    
}

```

### Class Skeleton

```java
//This service is used to communicate with openexchangerates.org to take historical data for the last 30 days of data 
//It uses OpenExchangeService to take historical data via the getHistoricalExchangeRates method 
//The historical data is analyzed, and each currency's average exchange rate for the last month is calculated.
@Service
public class HistoricalRateService { 

 private OpenExchangeService openExchangeService; 

 @Autowired 
 public HistoricalRateService(OpenExchangeService openExchangeService) { 
        this.openExchangeService = openExchangeService; 
 }
```

### Mind Limitations !!!

+ Always review the generated code to ensure it meets the project requirements. Copilot has some limitations as a code completion (inline) tool:
+ Don't expect Copilot to generate an entire project structure automatically.
+ Do not rely on autocomplete without first understanding the context.
+ Don't forget about manual version verification and specific project requirements.
+ Don't expect Copilot to always understand your project's requirements without direction.
+ Don't expect Copilot to always be updated with the most recent versions or changes.