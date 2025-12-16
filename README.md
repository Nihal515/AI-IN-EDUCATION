# 📊 Educational Institution Analytics Dashboard

A comprehensive, interactive dashboard for analyzing student and faculty performance metrics with AI usage tracking, built using Streamlit and Plotly.

---

## 🌟 Features

### **Multi-Page Dashboard**
- **Executive Overview**: High-level KPIs and department performance scorecards
- **Student Analytics**: Detailed student performance, attendance, AI usage, and engagement analysis
- **Faculty Analytics**: Faculty efficiency, AI dependency, teaching hours, and experience analysis
- **Comparative Analysis**: Cross-functional insights comparing students and faculty metrics

### **Key Capabilities**
✅ **Interactive Visualizations**: 12+ dynamic charts with vibrant, professional color schemes
✅ **Smart Flashcards**: 2 flashcards per visualization (Statistical Analysis + Business Insights)
✅ **Global & Local Filters**: Department, month, aggregation method, and page-specific filters
✅ **Anomaly Detection**: Automatic flagging of:
  - Students with <70% attendance
  - Scores below 40
  - Faculty with AI dependency >1.0
✅ **Real-time Filtering**: Auto-update all visualizations when filters change
✅ **Performance Thresholds**: Color-coded metrics (Good >75, Average 60-75, Poor <60)

---

## 📁 Project Structure

```
project/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── data/
    ├── faculty_master.csv         # Faculty master data
    ├── faculty_yearly_metrics.csv # Faculty monthly metrics
    ├── students_master.csv        # Student master data
    └── student_yearly_metrics.csv # Student monthly metrics
```

---

## 🚀 Installation & Setup

### **Prerequisites**
- Python 3.8 or higher
- pip (Python package manager)

### **Step 1: Clone or Download**
Download all files to your local directory.

### **Step 2: Install Dependencies**
Open terminal/command prompt in the project directory and run:

```bash
pip install -r requirements.txt
```

### **Step 3: Prepare Data**
Ensure your CSV files are in the correct location:
- Place all 4 CSV files in `/mnt/user-data/uploads/` directory
- OR update file paths in `app.py` (lines 107-110) to match your data location

### **Step 4: Run the Dashboard**
```bash
streamlit run app.py
```

The dashboard will automatically open in your default web browser at `http://localhost:8501`

---

## 📊 Data Schema

### **Faculty Master (faculty_master.csv)**
| Column | Description |
|--------|-------------|
| faculty_id | Unique faculty identifier |
| department | Department name (Computer Science, IT, Data Science, AI) |
| designation | Job title (Professor, Associate Professor, Assistant Professor) |
| experience_years | Years of teaching experience |
| baseline_work_hours | Standard weekly work hours |

### **Faculty Yearly Metrics (faculty_yearly_metrics.csv)**
| Column | Description |
|--------|-------------|
| faculty_id | Foreign key to faculty_master |
| month | Month name (Jan-Dec) |
| teaching_hours | Hours spent teaching |
| ai_usage_hours | Hours using AI tools |
| ai_dependency_ratio | AI hours / Teaching hours |
| student_avg_score | Average score of students taught |
| efficiency_score | Overall efficiency metric (0-100) |

### **Students Master (students_master.csv)**
| Column | Description |
|--------|-------------|
| student_id | Unique student identifier |
| degree_level | UG (Undergraduate) or PG (Postgraduate) |
| department | Department name |
| year_of_study | Academic year (1-4) |
| baseline_score | Initial performance baseline |
| baseline_attendance | Initial attendance baseline |

### **Student Yearly Metrics (student_yearly_metrics.csv)**
| Column | Description |
|--------|-------------|
| student_id | Foreign key to students_master |
| month | Month name (Jan-Dec) |
| attendance_pct | Monthly attendance percentage |
| avg_score | Average score for the month |
| ai_usage_hours | Hours spent using AI tools |
| performance_delta | Change from baseline score |
| engagement_score | Overall engagement metric (0-100) |

---

## 🎯 How to Use the Dashboard

### **Navigation**
1. **Sidebar Menu**: Select between 4 main pages
2. **Global Filters**: Apply to entire dashboard
   - Department selection
   - Month range
   - Aggregation method (Mean, Latest, Sum)
3. **Page-Specific Filters**: Additional local filters per page

### **Page Guide**

#### 📌 **Executive Overview**
- View institution-wide KPIs
- Department performance radar chart
- Monthly trends for students and faculty
- Ideal for: C-suite, board meetings, quarterly reviews

#### 🎓 **Student Analytics**
**Local Filters**: Degree Level, Year of Study, Score Range

**Visualizations**:
1. Attendance vs Performance scatter plot with trend line
2. AI Usage distribution by department (box plot)
3. Performance trends over time (line chart)
4. Engagement score analysis by year and degree

**Use Cases**: Student support, curriculum planning, intervention targeting

#### 👨‍🏫 **Faculty Analytics**
**Local Filters**: Designation, Experience Range

**Visualizations**:
1. AI Dependency vs Efficiency (scatter with correlation)
2. Teaching Hours vs AI Usage trends
3. Performance by designation (grouped bars)
4. Experience vs Performance analysis

**Use Cases**: Faculty development, hiring decisions, resource allocation

#### 🔀 **Comparative Analysis**
**Features**:
1. Student vs Faculty AI usage comparison
2. Correlation matrix for all metrics
3. Department performance scorecard

**Use Cases**: Strategic planning, cross-functional insights, benchmarking

---

## 📈 Understanding Flashcards

Each visualization includes **2 flashcards** displayed side-by-side:

### **📊 Statistical Analysis Flashcard (Pink/Red)**
- Mean, median, standard deviation
- Outlier detection
- Correlation coefficients
- Anomaly counts
- Data ranges and distributions

### **💡 Business Insights Flashcard (Blue)**
- Actionable recommendations
- Strategic implications
- Risk identification
- Best practice suggestions
- ROI and impact analysis

---

## 🎨 Color Schemes

### **Student Visualizations**
- Primary: Blues, Greens (#3498db, #66c2a5)
- Accent: Set2, Set3 palettes
- Purpose: Calm, trustworthy, academic

### **Faculty Visualizations**
- Primary: Oranges, Purples (#ff7f0e, #9467bd)
- Accent: Warm tones, Safe palette
- Purpose: Professional, authoritative, energetic

### **Performance Categories**
- 🟢 Good (>75): Green
- 🟡 Average (60-75): Yellow/Orange
- 🔴 Poor (<60): Red

---

## 🔧 Customization Guide

### **Changing Performance Thresholds**
Edit the `get_performance_category()` function in `app.py` (lines 236-242):
```python
def get_performance_category(score):
    if score >= 75:  # Change this
        return "Good"
    elif score >= 60:  # Change this
        return "Average"
    else:
        return "Poor"
```

### **Adding New Visualizations**
1. Create a new plotting function following the pattern:
   ```python
   def plot_your_visualization(data_dict, filters):
       st.subheader("Your Title")
       # ... your code ...
       st.plotly_chart(fig, use_container_width=True)
       
       # Add flashcards
       col1, col2 = st.columns(2)
       with col1:
           st.markdown(create_flashcard("Stats", stats_content, "stats"))
       with col2:
           st.markdown(create_flashcard("Insights", insights_content, "insights"))
   ```

2. Call it in the appropriate page section in `main()` function

### **Modifying Color Schemes**
Update color palettes in plotting functions:
- `color_discrete_sequence` parameter in Plotly Express
- `marker_color` parameter in Plotly Graph Objects
- CSS gradient colors in the `<style>` section (lines 39-156)

### **Adjusting Anomaly Thresholds**
Edit sidebar anomaly detection (lines 1032-1044):
```python
low_attendance_count = len(student_metrics_filtered[student_metrics_filtered['attendance_pct'] < 70])  # Change 70
low_score_count = len(student_metrics_filtered[student_metrics_filtered['avg_score'] < 40])  # Change 40
high_ai_dependency_count = len(faculty_metrics_filtered[faculty_metrics_filtered['ai_dependency_ratio'] > 1.0])  # Change 1.0
```

---

## 🐛 Troubleshooting

### **Dashboard won't start**
- Verify Python version: `python --version` (should be 3.8+)
- Reinstall dependencies: `pip install -r requirements.txt --upgrade`
- Check port availability: Try `streamlit run app.py --server.port 8502`

### **Data not loading**
- Verify CSV file paths in lines 107-110 of `app.py`
- Check CSV file encoding (should be UTF-8)
- Ensure column names match exactly (case-sensitive)

### **Visualizations not updating**
- Clear Streamlit cache: Click "C" key or "Clear Cache" in menu
- Restart the application
- Check browser console for JavaScript errors

### **Performance issues**
- Reduce date range using month filter
- Limit department selection
- Use "Latest" aggregation instead of "Mean" for faster loading

---

## 📊 Sample Insights You Can Generate

### **For Academic Leaders**
- "Which department has the highest ROI on AI investment?"
- "What's the correlation between faculty experience and student outcomes?"
- "Where should we focus retention efforts?"

### **For Faculty Development**
- "Which teaching methods correlate with engagement?"
- "How does AI adoption impact efficiency?"
- "What's the optimal AI-to-teaching hour ratio?"

### **For Student Success**
- "Which students are at risk of dropping out?"
- "What engagement patterns predict performance?"
- "How does AI usage differ across degree levels?"

---

## 🔒 Data Privacy & Security

- All data is processed locally on your machine
- No external API calls or data transmission
- CSV files are cached in memory using Streamlit's `@st.cache_data`
- Suitable for sensitive institutional data

---

## 🚀 Performance Optimization

### **For Large Datasets**
1. **Use Latest Aggregation**: Faster than calculating means
2. **Apply Filters Early**: Reduce data volume before visualization
3. **Limit Month Range**: Focus on recent terms
4. **Enable Streamlit Caching**: Already implemented via `@st.cache_data`

### **Deployment Considerations**
- For production: Use Streamlit Cloud, Heroku, or AWS
- Enable authentication for sensitive data
- Set up regular data refresh pipelines
- Consider database backend for very large datasets

---

## 📝 Version History

### **v1.0.0** - Current Release
- Initial release with 4 pages
- 12+ interactive visualizations
- 24+ flashcards (2 per chart)
- Global and local filtering
- Anomaly detection
- Mobile-responsive design

---

## 🤝 Support & Contribution

### **Getting Help**
- Check this README thoroughly
- Review inline code comments in `app.py`
- Test with sample data first

### **Feature Requests**
Consider adding:
- Export functionality (CSV, PDF)
- Predictive analytics (forecasting)
- Email alerts for anomalies
- Custom report generation
- Integration with LMS systems

---

## 📄 License

This dashboard is provided as-is for educational and analytical purposes.

---

## 🎉 Quick Start Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] CSV files in correct location
- [ ] Dashboard starts successfully (`streamlit run app.py`)
- [ ] All 4 pages load without errors
- [ ] Filters work correctly
- [ ] Flashcards display properly
- [ ] Anomaly alerts show in sidebar

---

**Built with ❤️ using Streamlit, Plotly, and Pandas**

*For questions or issues, please refer to the troubleshooting section above.*
