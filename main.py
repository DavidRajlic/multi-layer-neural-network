import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math

class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.Weight1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.Weight2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
    
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x)) # value between 0 and  1
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X): 
        
        self.z1 = np.dot(X, self.Weight1) 
        self.a1 = self.sigmoid(self.z1) #aktivacija št od 0 do 1, ki greod v naslendjo plast
        self.z2 = np.dot(self.a1, self.Weight2)
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
       
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)
        
        self.hidden_error = self.output_delta.dot(self.Weight2.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.a1)
        
        self.Weight2 += self.a1.T.dot(self.output_delta) * self.learning_rate
        self.Weight1 += X.T.dot(self.hidden_delta) * self.learning_rate
    
    def train(self, X, y, epochs, error_threshold=0.01):
        errors = []
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                input_data = X[i].reshape(1, -1)
                target = y[i].reshape(1, -1)
                
                output = self.forward(input_data)
             
                error = (target - output) ** 2
                total_error += np.mean(error)
            
                self.backward(input_data, target, output)
            
            avg_error = total_error / len(X)
            errors.append(avg_error)
            
            if avg_error <= error_threshold:
                print(f"Training stopped at epoch {epoch+1} with error: {avg_error}")
                break
                
            if epoch % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Error: {avg_error}")
        
        return errors
    
    def predict(self, X):
        output = self.forward(X)
        return output

class Symbol:
    def __init__(self, points, label):
        self.points = points
        self.label = label
        self.vector = None
        
class SymbolRecognition:
    def __init__(self, root):
        self.root = root
        self.root.title("Symbol Recognition")
        self.root.geometry("1000x800")
        
        self.neural_network = None
        self.symbols = []
        self.current_points = []
        self.unique_labels = set()
        self.drawing = False
        self.training_mode = True
        self.vector_size = 10  # Default value for N
        
        self.create_ui()
        
    def create_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        param_frame = ttk.LabelFrame(left_panel, text="Neural Network Parameters")
        param_frame.pack(fill=tk.X, pady=10)
        ttk.Label(param_frame, text="Vector size (N):").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        self.vector_size_var = tk.IntVar(value=10)
        ttk.Spinbox(param_frame, from_=5, to=50, textvariable=self.vector_size_var, width=10).grid(column=1, row=0, padx=5, pady=5)
        ttk.Label(param_frame, text="Hidden neurons:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        self.hidden_neurons_var = tk.IntVar(value=20)
        ttk.Spinbox(param_frame, from_=5, to=100, textvariable=self.hidden_neurons_var, width=10).grid(column=1, row=1, padx=5, pady=5)
        ttk.Label(param_frame, text="Learning rate:").grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)
        self.learning_rate_var = tk.DoubleVar(value=0.1)
        ttk.Spinbox(param_frame, from_=0.001, to=1.0, increment=0.01, textvariable=self.learning_rate_var, width=10).grid(column=1, row=2, padx=5, pady=5)
        ttk.Label(param_frame, text="Error thrshold:").grid(column=0, row=3, sticky=tk.W, padx=5, pady=5)
        self.error_threshold_var = tk.DoubleVar(value=0.01)
        ttk.Spinbox(param_frame, from_=0.0001, to=0.1, increment=0.001, textvariable=self.error_threshold_var, width=10).grid(column=1, row=3, padx=5, pady=5)
        ttk.Label(param_frame, text="Max epochs:").grid(column=0, row=4, sticky=tk.W, padx=5, pady=5)
        self.epochs_var = tk.IntVar(value=1000)
        ttk.Spinbox(param_frame, from_=100, to=10000, increment=100, textvariable=self.epochs_var, width=10).grid(column=1, row=4, padx=5, pady=5)
        
        symbol_frame = ttk.LabelFrame(left_panel, text="Symbol Input")
        symbol_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(symbol_frame, text="Symbol label:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        self.symbol_label_var = tk.StringVar()
        ttk.Entry(symbol_frame, textvariable=self.symbol_label_var, width=10).grid(column=1, row=0, padx=5, pady=5)
        
        ttk.Button(symbol_frame, text="Add Symbol", command=self.add_symbol).grid(column=0, row=1, padx=5, pady=5)
        ttk.Button(symbol_frame, text="Clear", command=self.clear_canvas).grid(column=1, row=1, padx=5, pady=5)
    
        symbols_list_frame = ttk.LabelFrame(left_panel, text="Training Symbols")
        symbols_list_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.symbols_listbox = tk.Listbox(symbols_list_frame, height=10)
        self.symbols_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        control_frame = ttk.Frame(left_panel)
        control_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(control_frame, text="Train Network", command=self.train_network).grid(column=0, row=0, padx=5, pady=5)
        self.mode_var = tk.StringVar(value="Training")
        ttk.Label(control_frame, text="Mode:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        self.mode_label = ttk.Label(control_frame, textvariable=self.mode_var, foreground="blue")
        self.mode_label.grid(column=1, row=1, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Turn recognition mode on", command=self.toggle_mode).grid(column=0, row=2, columnspan=2, padx=5, pady=5)
        
        canvas_frame = ttk.LabelFrame(right_panel, text="Drawing Canvas")
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.canvas = tk.Canvas(canvas_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas.bind("<ButtonPress-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        
        # Result of recognition
        result_frame = ttk.LabelFrame(right_panel, text="Recognition Result")
        result_frame.pack(fill=tk.X, pady=10)
        
        self.result_var = tk.StringVar(value="Draw a symbol to recognize")
        ttk.Label(result_frame, textvariable=self.result_var, font=("Arial", 12)).pack(padx=5, pady=5)
        
        # Graph for errors
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.graph_canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.graph_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax.set_title("Training Error")
        self.ax.set_xlabel("Epochs")
        self.ax.set_ylabel("Error")
        self.graph_canvas.draw()
        
    def start_drawing(self, event):
        self.drawing = True
        self.current_points = [(event.x, event.y)]
        self.canvas.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, fill="black")
        
    def draw(self, event):
        if self.drawing:
            self.current_points.append((event.x, event.y))
            x1, y1 = self.current_points[-2] # befor last point
            x2, y2 = self.current_points[-1] #last point 
            self.canvas.create_line(x1, y1, x2, y2, fill="black", width=2)
            self.canvas.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, fill="black")
            
    def stop_drawing(self, event):
        self.drawing = False
        if len(self.current_points) < 2:
            return
        
        # If recognition mode is ON recognize the symbol
        if not self.training_mode and self.neural_network is not None:
            self.recognize_symbol()
            
    def clear_canvas(self):
        self.canvas.delete("all")
        self.current_points = []
        self.result_var.set("Draw a symbol")
        
    def add_symbol(self):
        if not self.current_points or not self.training_mode:
            return
            
        label = self.symbol_label_var.get().strip()
        if not label:
            messagebox.showerror("Error", "Please enter a symbol label")
            return
            
        # Add new symbol
        symbol = Symbol(self.current_points.copy(), label)
        self.unique_labels.add(label)
        self.symbols.append(symbol)
        
        self.symbols_listbox.insert(tk.END, f"{label} (points: {len(self.current_points)})")
        self.clear_canvas()
        
    def preprocess_symbol(self, points, n_vectors):
        if len(points) < 2:
            return None
            
        points = points.copy()
        
        fixed_points = [points[0], points[-1]]
        middle_points = points[1:-1]
        
        # Reduce to points (N+1)
        while len(middle_points) > n_vectors - 1:
            # Search the closest pair
            min_dist = float('inf')
            min_idx = -1
            
            for i in range(len(middle_points) - 1):
                p1 = middle_points[i]
                p2 = middle_points[i + 1]
                dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
                    
            if min_idx >= 0:
                p1 = middle_points[min_idx]
                p2 = middle_points[min_idx + 1]
                midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                middle_points[min_idx] = midpoint
                middle_points.pop(min_idx + 1)
                
        processed_points = [fixed_points[0]] + middle_points + [fixed_points[1]]
        
        # Converting to vectors and normalizing
        vectors = []
        for i in range(len(processed_points) - 1):
            p1 = processed_points[i]
            p2 = processed_points[i + 1]
            
            vectorX = p2[0] - p1[0]
            vectorY= p2[1] - p1[1]
        
            magnitude = math.sqrt(vectorX**2 + vectorY**2)
            
            # Normalization
            if magnitude > 0:
                vectorX = vectorX / magnitude
                vectorY = vectorY / magnitude
                
            vectors.extend([vectorX, vectorY])
            
        return np.array(vectors)
    
    def train_network(self):
        if len(self.symbols) == 0:
            messagebox.showerror("Error", "No symbols to train on")
            return

        N = self.vector_size_var.get()
        hidden_neurons = self.hidden_neurons_var.get()
        learning_rate = self.learning_rate_var.get()
        error_threshold = self.error_threshold_var.get()
        epochs = self.epochs_var.get()
        
        X = []
        labels = []
        
        for symbol in self.symbols:
            vector = self.preprocess_symbol(symbol.points, N)
            if vector is not None:
                symbol.vector = vector
                X.append(vector)
                labels.append(symbol.label)
        
        unique_labels = sorted(list(self.unique_labels))
        num_classes = len(unique_labels)
        print(num_classes)
        
        self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        self.idx_to_label = {i: label for i, label in enumerate(unique_labels)}

        Y = []
        for label in labels:
            one_hot = np.zeros(num_classes)
            one_hot[self.label_to_idx[label]] = 1
            Y.append(one_hot)
        
        X = np.array(X)
        Y = np.array(Y)
        
        # Create and train the neural network
        input_size = 2 * N
        hidden_size = hidden_neurons
        output_size = num_classes
        
        self.neural_network = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

        print("X shape:", X)
        print("y shape:", Y)
        
        # Train the network
        errors = self.neural_network.train(X, Y, epochs, error_threshold)
        
        self.ax.clear()
        self.ax.plot(errors)
        self.ax.set_title("Training Error")
        self.ax.set_xlabel("Epochs")
        self.ax.set_ylabel("Error")
        self.graph_canvas.draw()
        
        messagebox.showinfo("Training Complete", "Neural Network training completed!")
        
    def toggle_mode(self):
        self.training_mode = not self.training_mode
        
        if self.training_mode:
            self.mode_var.set("Training")
            self.mode_label.config(foreground="blue")
        else:
            if self.neural_network is None:
                messagebox.showerror("Error", "Please train the network before recognition")
                self.training_mode = True
                return
                
            self.mode_var.set("Recognition")
            self.mode_label.config(foreground="red")
                
    def recognize_symbol(self):
        if not self.current_points or self.neural_network is None:
            return
        
        n = self.vector_size_var.get()
        vector = self.preprocess_symbol(self.current_points, n)
        
        if vector is not None:
            output = self.neural_network.predict(vector.reshape(1, -1))
            predicted_idx = np.argmax(output)
            predicted_label = self.idx_to_label[predicted_idx]
            confidence = output[0][predicted_idx]
            
            self.result_var.set(f"Recognized: {predicted_label} (Confidence: {confidence:.2f})")
        
if __name__ == "__main__":
    root = tk.Tk()
    app = SymbolRecognition(root)
    root.mainloop()