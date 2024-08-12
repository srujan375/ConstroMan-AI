# Analysis/dashboard_generator.py

import matplotlib.pyplot as plt
import json

class DashboardPlotter:
    def __init__(self, json_data):
        self.data = json.loads(json_data) if isinstance(json_data, str) else json_data
        self.dashboard = self.data["Dashboard"]

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 7))  # Increased figure size to accommodate legend
        chart_type = self.dashboard["Type"]
        
        try:
            if chart_type in ["LineChart", "BarChart", "ScatterPlot"]:
                self._plot_xy_chart(ax, chart_type)
            elif chart_type in ["PieChart", "DonutChart"]:
                self._plot_pie_chart(ax, chart_type)
            elif chart_type == "Histogram":
                self._plot_histogram(ax)
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")

            ax.set_title(self.dashboard["Name"])
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error plotting chart: {str(e)}")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error plotting chart: {str(e)}", ha='center', va='center')
            return fig

    def _plot_xy_chart(self, ax, chart_type):
        x_data = self.dashboard.get("X-axis data", [])
        y_data = self.dashboard.get("Y-axis data", [])
        
        if len(x_data) != len(y_data):
            raise ValueError("X-axis and Y-axis data must have the same length")
        
        if chart_type == "LineChart":
            ax.plot(x_data, y_data, label=self.dashboard.get("Y-axis label", "Data"))
            # Format y-axis labels
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        if chart_type == "BarChart":
            ax.bar(x_data, y_data, label=self.dashboard.get("Y-axis label", "Data"))
            # Format y-axis labels
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        elif chart_type == "ScatterPlot":
            ax.scatter(x_data, y_data, label=self.dashboard.get("Y-axis label", "Data"))
        
        ax.set_xlabel(self.dashboard.get("X-axis label", ""))
        ax.set_ylabel(self.dashboard.get("Y-axis label", ""))
        if x_data:
            ax.set_xticklabels(x_data, rotation=45, ha="right")
        
        ax.legend(loc='upper right')

    def _plot_pie_chart(self, ax, chart_type):
        labels = self.dashboard.get("Labels", [])
        values = self.dashboard.get("Values", [])
        
        if len(labels) != len(values):
            raise ValueError("Number of labels must match number of values")
        
        if not labels or not values:
            raise ValueError("Both labels and values must be provided for pie chart")
        
        if chart_type == "DonutChart":
            wedges, texts, autotexts = ax.pie(values, labels=None, autopct='%1.1f%%', pctdistance=0.85)
            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
            ax.add_artist(centre_circle)
        else:  # PieChart
            wedges, texts, autotexts = ax.pie(values, labels=None, autopct='%1.1f%%')
        
        ax.legend(wedges, labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    def _plot_histogram(self, ax):
        data = self.dashboard.get("Values", [])
        if not data:
            raise ValueError("No data provided for histogram")
        
        n, bins, patches = ax.hist(data, bins='auto', label=self.dashboard.get("X-axis label", "Data"))
        ax.set_xlabel(self.dashboard.get("X-axis label", ""))
        ax.set_ylabel(self.dashboard.get("Y-axis label", ""))
        ax.legend(loc='upper right')