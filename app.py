"""
Educational Institution Analytics Dashboard
A comprehensive Streamlit dashboard for analyzing student and faculty performance metrics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ================================
# PAGE CONFIGURATION
# ================================
st.set_page_config(
    page_title="Education Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CUSTOM CSS FOR STYLING
# ================================
st.markdown("""
    <style>
    /* Main title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Flashcard styling */
    .flashcard {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
        color: white;
    }
    
    .flashcard-stats {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
        color: white;
    }
    
    .flashcard-insights {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
        color: white;
    }
    
    .flashcard h4 {
        margin-top: 0;
        font-size: 1.2rem;
        border-bottom: 2px solid rgba(255,255,255,0.3);
        padding-bottom: 10px;
    }
    
    .flashcard-stats h4 {
        margin-top: 0;
        font-size: 1.2rem;
        border-bottom: 2px solid rgba(255,255,255,0.3);
        padding-bottom: 10px;
    }
    
    .flashcard-insights h4 {
        margin-top: 0;
        font-size: 1.2rem;
        border-bottom: 2px solid rgba(255,255,255,0.3);
        padding-bottom: 10px;
    }
    
    /* KPI Card styling */
    .kpi-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 5px solid #1f77b4;
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    
    .kpi-label {
        font-size: 1rem;
        color: #666;
        margin-top: 5px;
    }
    
    /* Metric container */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Anomaly badge */
    .anomaly-badge {
        background: #ff4444;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 0.9rem;
        display: inline-block;
        margin: 5px;
    }
    
    .warning-badge {
        background: #ffaa00;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 0.9rem;
        display: inline-block;
        margin: 5px;
    }
    
    .success-badge {
        background: #00C851;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 0.9rem;
        display: inline-block;
        margin: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# ================================
# DATA LOADING AND CACHING
# ================================
@st.cache_data
def load_data():
    """Load all CSV files and return as dictionary"""
    try:
        faculty_master = pd.read_csv('/mnt/user-data/uploads/faculty_master.csv')
        faculty_metrics = pd.read_csv('/mnt/user-data/uploads/faculty_yearly_metrics.csv')
        students_master = pd.read_csv('/mnt/user-data/uploads/students_master.csv')
        student_metrics = pd.read_csv('/mnt/user-data/uploads/student_yearly_metrics.csv')
        
        # Convert month names to ordered categorical for proper sorting
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        faculty_metrics['month'] = pd.Categorical(faculty_metrics['month'], 
                                                   categories=month_order, ordered=True)
        student_metrics['month'] = pd.Categorical(student_metrics['month'], 
                                                   categories=month_order, ordered=True)
        
        return {
            'faculty_master': faculty_master,
            'faculty_metrics': faculty_metrics,
            'students_master': students_master,
            'student_metrics': student_metrics
        }
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# ================================
# HELPER FUNCTIONS
# ================================

def create_flashcard(title, content, card_type="stats"):
    """Create a styled flashcard for displaying information"""
    card_class = "flashcard-stats" if card_type == "stats" else "flashcard-insights"
    icon = "📊" if card_type == "stats" else "💡"
    
    html = f"""
    <div class="{card_class}">
        <h4>{icon} {title}</h4>
        <div style='font-size: 0.95rem; line-height: 1.6;'>
            {content}
        </div>
    </div>
    """
    return html

def detect_outliers(data, column):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def get_performance_category(score):
    """Categorize performance based on score"""
    if score >= 75:
        return "Good"
    elif score >= 60:
        return "Average"
    else:
        return "Poor"

def apply_filters(data, filters):
    """Apply multiple filters to a dataframe"""
    filtered_data = data.copy()
    
    for column, values in filters.items():
        if values and len(values) > 0:
            filtered_data = filtered_data[filtered_data[column].isin(values)]
    
    return filtered_data

# ================================
# VISUALIZATION FUNCTIONS
# ================================

def create_kpi_cards(data_dict, filters):
    """Create KPI cards for the overview page"""
    
    # Apply filters to student and faculty data
    students = apply_filters(data_dict['students_master'], filters)
    faculty = apply_filters(data_dict['faculty_master'], filters)
    
    # Merge with metrics
    student_metrics_filtered = data_dict['student_metrics'][
        data_dict['student_metrics']['student_id'].isin(students['student_id'])
    ]
    faculty_metrics_filtered = data_dict['faculty_metrics'][
        data_dict['faculty_metrics']['faculty_id'].isin(faculty['faculty_id'])
    ]
    
    # Calculate KPIs
    total_students = len(students)
    total_faculty = len(faculty)
    avg_student_score = student_metrics_filtered['avg_score'].mean()
    avg_attendance = student_metrics_filtered['attendance_pct'].mean()
    avg_faculty_efficiency = faculty_metrics_filtered['efficiency_score'].mean()
    avg_ai_dependency = faculty_metrics_filtered['ai_dependency_ratio'].mean()
    
    # Display KPIs
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Students", f"{total_students:,}", 
                  delta=None, delta_color="normal")
    
    with col2:
        st.metric("Total Faculty", f"{total_faculty:,}", 
                  delta=None, delta_color="normal")
    
    with col3:
        st.metric("Avg Student Score", f"{avg_student_score:.1f}", 
                  delta=None, delta_color="normal")
    
    with col4:
        st.metric("Avg Attendance", f"{avg_attendance:.1f}%", 
                  delta=None, delta_color="normal")
    
    with col5:
        st.metric("Avg Faculty Efficiency", f"{avg_faculty_efficiency:.1f}", 
                  delta=None, delta_color="normal")
    
    with col6:
        st.metric("Avg AI Dependency", f"{avg_ai_dependency:.2f}", 
                  delta=None, delta_color="normal")

def plot_student_attendance_vs_performance(data_dict, filters):
    """Create scatter plot of student attendance vs performance with flashcards"""
    
    st.subheader("📈 Student Attendance vs Performance Analysis")
    
    # Prepare data
    students = apply_filters(data_dict['students_master'], filters)
    student_metrics = data_dict['student_metrics'][
        data_dict['student_metrics']['student_id'].isin(students['student_id'])
    ]
    
    # Merge with master data
    plot_data = student_metrics.merge(students, on='student_id')
    
    # Aggregate by aggregation method
    agg_method = st.session_state.get('aggregation_method', 'mean')
    
    if agg_method == 'mean':
        agg_data = plot_data.groupby('student_id').agg({
            'attendance_pct': 'mean',
            'avg_score': 'mean',
            'department': 'first',
            'degree_level': 'first'
        }).reset_index()
    elif agg_method == 'latest':
        agg_data = plot_data.sort_values('month').groupby('student_id').tail(1)
    else:  # sum
        agg_data = plot_data.groupby('student_id').agg({
            'attendance_pct': 'mean',
            'avg_score': 'mean',
            'department': 'first',
            'degree_level': 'first'
        }).reset_index()
    
    # Add performance category
    agg_data['Performance'] = agg_data['avg_score'].apply(get_performance_category)
    
    # Create scatter plot with vibrant colors
    fig = px.scatter(
        agg_data, 
        x='attendance_pct', 
        y='avg_score',
        color='department',
        size='avg_score',
        hover_data=['student_id', 'degree_level', 'Performance'],
        title='Student Attendance vs Average Score',
        labels={'attendance_pct': 'Attendance (%)', 'avg_score': 'Average Score'},
        color_discrete_sequence=px.colors.qualitative.Set2,
        template='plotly_white'
    )
    
    # Add trend line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        agg_data['attendance_pct'], agg_data['avg_score']
    )
    
    x_range = np.array([agg_data['attendance_pct'].min(), agg_data['attendance_pct'].max()])
    y_range = slope * x_range + intercept
    
    fig.add_trace(go.Scatter(
        x=x_range, 
        y=y_range,
        mode='lines',
        name=f'Trend (R²={r_value**2:.3f})',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig.update_layout(
        height=500,
        showlegend=True,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate statistics for flashcards
    outliers, lower, upper = detect_outliers(agg_data, 'avg_score')
    correlation = agg_data[['attendance_pct', 'avg_score']].corr().iloc[0, 1]
    low_attendance = len(agg_data[agg_data['attendance_pct'] < 70])
    poor_performers = len(agg_data[agg_data['avg_score'] < 40])
    
    # Create flashcards
    col1, col2 = st.columns(2)
    
    with col1:
        stats_content = f"""
        <b>Statistical Summary:</b><br>
        • Mean Score: {agg_data['avg_score'].mean():.2f}<br>
        • Mean Attendance: {agg_data['attendance_pct'].mean():.2f}%<br>
        • Correlation: {correlation:.3f}<br>
        • Outliers Detected: {len(outliers)} students<br>
        • Low Attendance (<70%): <span class='anomaly-badge'>{low_attendance} students</span><br>
        • Poor Performance (<40): <span class='anomaly-badge'>{poor_performers} students</span><br>
        • Score Range: {agg_data['avg_score'].min():.1f} - {agg_data['avg_score'].max():.1f}
        """
        st.markdown(create_flashcard("Statistical Analysis", stats_content, "stats"), 
                    unsafe_allow_html=True)
    
    with col2:
        insights_content = f"""
        <b>Key Business Insights:</b><br>
        • <b>Weak Correlation Alert:</b> Attendance has surprisingly low correlation ({correlation:.3f}) with performance<br>
        • <b>Action Required:</b> {low_attendance} students need immediate attendance intervention<br>
        • <b>Academic Support:</b> {poor_performers} students scoring below 40 require tutoring<br>
        • <b>Recommendation:</b> Focus on engagement quality over attendance quantity<br>
        • <b>Top Performers:</b> {len(agg_data[agg_data['avg_score'] > 85])} students excel despite varying attendance patterns
        """
        st.markdown(create_flashcard("Business Insights", insights_content, "insights"), 
                    unsafe_allow_html=True)

def plot_student_ai_usage_distribution(data_dict, filters):
    """Create distribution plot of student AI usage with flashcards"""
    
    st.subheader("🤖 Student AI Usage Distribution by Department")
    
    # Prepare data
    students = apply_filters(data_dict['students_master'], filters)
    student_metrics = data_dict['student_metrics'][
        data_dict['student_metrics']['student_id'].isin(students['student_id'])
    ]
    
    # Merge with master data
    plot_data = student_metrics.merge(students, on='student_id')
    
    # Aggregate
    agg_method = st.session_state.get('aggregation_method', 'mean')
    if agg_method == 'mean':
        agg_data = plot_data.groupby(['department', 'student_id']).agg({
            'ai_usage_hours': 'mean'
        }).reset_index()
    elif agg_method == 'latest':
        agg_data = plot_data.sort_values('month').groupby(['department', 'student_id']).tail(1)
    else:
        agg_data = plot_data.groupby(['department', 'student_id']).agg({
            'ai_usage_hours': 'sum'
        }).reset_index()
    
    # Create box plot with vibrant colors
    fig = px.box(
        agg_data, 
        x='department', 
        y='ai_usage_hours',
        color='department',
        title='AI Usage Hours Distribution by Department',
        labels={'ai_usage_hours': 'AI Usage Hours', 'department': 'Department'},
        color_discrete_sequence=px.colors.qualitative.Set3,
        template='plotly_white'
    )
    
    fig.update_layout(
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate statistics
    dept_stats = agg_data.groupby('department')['ai_usage_hours'].agg(['mean', 'median', 'std', 'min', 'max'])
    high_usage = len(agg_data[agg_data['ai_usage_hours'] > 30])
    low_usage = len(agg_data[agg_data['ai_usage_hours'] < 5])
    
    # Create flashcards
    col1, col2 = st.columns(2)
    
    with col1:
        stats_content = f"""
        <b>Statistical Summary:</b><br>
        • Overall Mean: {agg_data['ai_usage_hours'].mean():.2f} hours<br>
        • Overall Median: {agg_data['ai_usage_hours'].median():.2f} hours<br>
        • Std Deviation: {agg_data['ai_usage_hours'].std():.2f}<br>
        • High Usage (>30h): <span class='warning-badge'>{high_usage} students</span><br>
        • Low Usage (<5h): {low_usage} students<br>
        • Range: {agg_data['ai_usage_hours'].min():.1f} - {agg_data['ai_usage_hours'].max():.1f} hours<br>
        <b>Department Leaders:</b><br>
        • {dept_stats['mean'].idxmax()}: {dept_stats['mean'].max():.2f}h average
        """
        st.markdown(create_flashcard("Statistical Analysis", stats_content, "stats"), 
                    unsafe_allow_html=True)
    
    with col2:
        insights_content = f"""
        <b>Key Business Insights:</b><br>
        • <b>AI Adoption Variance:</b> Significant variation across departments suggests inconsistent integration<br>
        • <b>Training Need:</b> {low_usage} students underutilizing AI tools - potential skill gap<br>
        • <b>Best Practice Sharing:</b> {dept_stats['mean'].idxmax()} department leads in AI adoption - share their methodology<br>
        • <b>Resource Allocation:</b> High usage students ({high_usage}) may need advanced AI workshops<br>
        • <b>Equity Concern:</b> Ensure all students have equal AI tool access and training
        """
        st.markdown(create_flashcard("Business Insights", insights_content, "insights"), 
                    unsafe_allow_html=True)

def plot_student_performance_trends(data_dict, filters):
    """Create line chart of student performance trends over time with flashcards"""
    
    st.subheader("📊 Student Performance Trends Over Time")
    
    # Prepare data
    students = apply_filters(data_dict['students_master'], filters)
    student_metrics = data_dict['student_metrics'][
        data_dict['student_metrics']['student_id'].isin(students['student_id'])
    ]
    
    # Merge with master data
    plot_data = student_metrics.merge(students, on='student_id')
    
    # Aggregate by month and department
    monthly_data = plot_data.groupby(['month', 'department']).agg({
        'avg_score': 'mean',
        'performance_delta': 'mean'
    }).reset_index()
    
    # Sort by month
    monthly_data = monthly_data.sort_values('month')
    
    # Create line chart with vibrant colors
    fig = px.line(
        monthly_data, 
        x='month', 
        y='avg_score',
        color='department',
        markers=True,
        title='Average Student Scores by Department (Monthly Trend)',
        labels={'avg_score': 'Average Score', 'month': 'Month'},
        color_discrete_sequence=px.colors.qualitative.Vivid,
        template='plotly_white'
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate statistics
    monthly_avg = plot_data.groupby('month')['avg_score'].mean()
    best_month = monthly_avg.idxmax()
    worst_month = monthly_avg.idxmin()
    improvement = monthly_avg.iloc[-1] - monthly_avg.iloc[0]
    best_dept = monthly_data.groupby('department')['avg_score'].mean().idxmax()
    
    # Create flashcards
    col1, col2 = st.columns(2)
    
    with col1:
        stats_content = f"""
        <b>Statistical Summary:</b><br>
        • Overall Average: {monthly_avg.mean():.2f}<br>
        • Best Month: {best_month} ({monthly_avg.max():.2f})<br>
        • Worst Month: {worst_month} ({monthly_avg.min():.2f})<br>
        • Year-over-Year Change: {improvement:+.2f} points<br>
        • Best Performing Dept: {best_dept}<br>
        • Volatility (Std): {monthly_avg.std():.2f}<br>
        • Trend: {'📈 Improving' if improvement > 0 else '📉 Declining' if improvement < 0 else '➡️ Stable'}
        """
        st.markdown(create_flashcard("Statistical Analysis", stats_content, "stats"), 
                    unsafe_allow_html=True)
    
    with col2:
        insights_content = f"""
        <b>Key Business Insights:</b><br>
        • <b>Seasonal Pattern:</b> {best_month} shows peak performance - analyze successful teaching methods<br>
        • <b>Intervention Point:</b> {worst_month} requires curriculum review and student support<br>
        • <b>Department Success:</b> {best_dept} outperforms - share best practices institution-wide<br>
        • <b>Trend Analysis:</b> {'Positive momentum sustained' if improvement > 2 else 'Marginal improvement' if improvement > 0 else 'Urgent remediation needed'}<br>
        • <b>Strategic Focus:</b> Maintain consistency while targeting low-performance months
        """
        st.markdown(create_flashcard("Business Insights", insights_content, "insights"), 
                    unsafe_allow_html=True)

def plot_student_engagement_analysis(data_dict, filters):
    """Create engagement score analysis with flashcards"""
    
    st.subheader("💪 Student Engagement Score Analysis")
    
    # Prepare data
    students = apply_filters(data_dict['students_master'], filters)
    student_metrics = data_dict['student_metrics'][
        data_dict['student_metrics']['student_id'].isin(students['student_id'])
    ]
    
    # Merge with master data
    plot_data = student_metrics.merge(students, on='student_id')
    
    # Aggregate
    agg_method = st.session_state.get('aggregation_method', 'mean')
    if agg_method == 'mean':
        agg_data = plot_data.groupby(['degree_level', 'year_of_study']).agg({
            'engagement_score': 'mean'
        }).reset_index()
    elif agg_method == 'latest':
        agg_data = plot_data.sort_values('month').groupby(['degree_level', 'year_of_study']).tail(1)
        agg_data = agg_data.groupby(['degree_level', 'year_of_study'])['engagement_score'].mean().reset_index()
    else:
        agg_data = plot_data.groupby(['degree_level', 'year_of_study']).agg({
            'engagement_score': 'mean'
        }).reset_index()
    
    # Create grouped bar chart
    fig = px.bar(
        agg_data,
        x='year_of_study',
        y='engagement_score',
        color='degree_level',
        barmode='group',
        title='Engagement Score by Year of Study and Degree Level',
        labels={'engagement_score': 'Engagement Score', 'year_of_study': 'Year of Study'},
        color_discrete_map={'UG': '#66c2a5', 'PG': '#8da0cb'},
        template='plotly_white'
    )
    
    fig.update_layout(
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate statistics
    ug_data = plot_data[plot_data['degree_level'] == 'UG']['engagement_score']
    pg_data = plot_data[plot_data['degree_level'] == 'PG']['engagement_score']
    high_engagement = len(plot_data[plot_data['engagement_score'] > 80])
    low_engagement = len(plot_data[plot_data['engagement_score'] < 50])
    
    # Create flashcards
    col1, col2 = st.columns(2)
    
    with col1:
        stats_content = f"""
        <b>Statistical Summary:</b><br>
        • UG Avg Engagement: {ug_data.mean():.2f}<br>
        • PG Avg Engagement: {pg_data.mean():.2f}<br>
        • Overall Average: {plot_data['engagement_score'].mean():.2f}<br>
        • High Engagement (>80): <span class='success-badge'>{high_engagement} records</span><br>
        • Low Engagement (<50): <span class='anomaly-badge'>{low_engagement} records</span><br>
        • Best Year: Year {agg_data.loc[agg_data['engagement_score'].idxmax(), 'year_of_study']}<br>
        • Engagement Gap: {abs(ug_data.mean() - pg_data.mean()):.2f} points
        """
        st.markdown(create_flashcard("Statistical Analysis", stats_content, "stats"), 
                    unsafe_allow_html=True)
    
    with col2:
        insights_content = f"""
        <b>Key Business Insights:</b><br>
        • <b>{'PG' if pg_data.mean() > ug_data.mean() else 'UG'} Leadership:</b> {'Postgraduate' if pg_data.mean() > ug_data.mean() else 'Undergraduate'} students show higher engagement<br>
        • <b>At-Risk Students:</b> {low_engagement} instances of low engagement need counseling<br>
        • <b>Best Practice:</b> Year {agg_data.loc[agg_data['engagement_score'].idxmax(), 'year_of_study']} curriculum driving engagement - replicate across years<br>
        • <b>Retention Risk:</b> Low engagement correlates with dropout - implement early warning system<br>
        • <b>Resource Optimization:</b> Target interactive programs for years with lower engagement
        """
        st.markdown(create_flashcard("Business Insights", insights_content, "insights"), 
                    unsafe_allow_html=True)

def plot_faculty_ai_efficiency(data_dict, filters):
    """Create faculty AI usage vs efficiency scatter plot with flashcards"""
    
    st.subheader("🎯 Faculty AI Dependency vs Efficiency Analysis")
    
    # Prepare data
    faculty = apply_filters(data_dict['faculty_master'], filters)
    faculty_metrics = data_dict['faculty_metrics'][
        data_dict['faculty_metrics']['faculty_id'].isin(faculty['faculty_id'])
    ]
    
    # Merge with master data
    plot_data = faculty_metrics.merge(faculty, on='faculty_id')
    
    # Aggregate
    agg_method = st.session_state.get('aggregation_method', 'mean')
    if agg_method == 'mean':
        agg_data = plot_data.groupby('faculty_id').agg({
            'ai_dependency_ratio': 'mean',
            'efficiency_score': 'mean',
            'department': 'first',
            'designation': 'first',
            'experience_years': 'first'
        }).reset_index()
    elif agg_method == 'latest':
        agg_data = plot_data.sort_values('month').groupby('faculty_id').tail(1)
    else:
        agg_data = plot_data.groupby('faculty_id').agg({
            'ai_dependency_ratio': 'mean',
            'efficiency_score': 'mean',
            'department': 'first',
            'designation': 'first',
            'experience_years': 'first'
        }).reset_index()
    
    # Create scatter plot
    fig = px.scatter(
        agg_data,
        x='ai_dependency_ratio',
        y='efficiency_score',
        color='department',
        size='experience_years',
        hover_data=['faculty_id', 'designation'],
        title='Faculty AI Dependency vs Efficiency Score',
        labels={'ai_dependency_ratio': 'AI Dependency Ratio', 'efficiency_score': 'Efficiency Score'},
        color_discrete_sequence=px.colors.qualitative.Safe,
        template='plotly_white'
    )
    
    # Add trend line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        agg_data['ai_dependency_ratio'], agg_data['efficiency_score']
    )
    
    x_range = np.array([agg_data['ai_dependency_ratio'].min(), agg_data['ai_dependency_ratio'].max()])
    y_range = slope * x_range + intercept
    
    fig.add_trace(go.Scatter(
        x=x_range, 
        y=y_range,
        mode='lines',
        name=f'Trend (R²={r_value**2:.3f})',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig.update_layout(
        height=500,
        showlegend=True,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate statistics
    correlation = agg_data[['ai_dependency_ratio', 'efficiency_score']].corr().iloc[0, 1]
    high_ai_dependency = len(agg_data[agg_data['ai_dependency_ratio'] > 1.0])
    low_efficiency = len(agg_data[agg_data['efficiency_score'] < 50])
    high_efficiency = len(agg_data[agg_data['efficiency_score'] > 70])
    
    # Create flashcards
    col1, col2 = st.columns(2)
    
    with col1:
        stats_content = f"""
        <b>Statistical Summary:</b><br>
        • Mean AI Dependency: {agg_data['ai_dependency_ratio'].mean():.2f}<br>
        • Mean Efficiency: {agg_data['efficiency_score'].mean():.2f}<br>
        • Correlation: <b>{correlation:.3f}</b> (Strong Positive!)<br>
        • High AI Dependency (>1.0): <span class='warning-badge'>{high_ai_dependency} faculty</span><br>
        • Low Efficiency (<50): {low_efficiency} faculty<br>
        • High Efficiency (>70): <span class='success-badge'>{high_efficiency} faculty</span><br>
        • R² Value: {r_value**2:.3f}
        """
        st.markdown(create_flashcard("Statistical Analysis", stats_content, "stats"), 
                    unsafe_allow_html=True)
    
    with col2:
        insights_content = f"""
        <b>Key Business Insights:</b><br>
        • <b>Strong ROI:</b> AI adoption shows {correlation:.3f} correlation with efficiency - invest in AI training!<br>
        • <b>AI Champions:</b> {high_ai_dependency} faculty exceed 100% AI dependency - leverage as mentors<br>
        • <b>Efficiency Gap:</b> {low_efficiency} faculty need AI upskilling programs urgently<br>
        • <b>Success Formula:</b> Higher AI usage = Higher efficiency scores consistently<br>
        • <b>Strategic Priority:</b> Scale AI tools institution-wide to boost overall productivity
        """
        st.markdown(create_flashcard("Business Insights", insights_content, "insights"), 
                    unsafe_allow_html=True)

def plot_faculty_teaching_hours_analysis(data_dict, filters):
    """Create faculty teaching hours vs AI usage analysis with flashcards"""
    
    st.subheader("⏰ Faculty Teaching Hours vs AI Usage")
    
    # Prepare data
    faculty = apply_filters(data_dict['faculty_master'], filters)
    faculty_metrics = data_dict['faculty_metrics'][
        data_dict['faculty_metrics']['faculty_id'].isin(faculty['faculty_id'])
    ]
    
    # Merge with master data
    plot_data = faculty_metrics.merge(faculty, on='faculty_id')
    
    # Aggregate by department and month
    monthly_data = plot_data.groupby(['month', 'department']).agg({
        'teaching_hours': 'mean',
        'ai_usage_hours': 'mean'
    }).reset_index()
    
    # Sort by month
    monthly_data = monthly_data.sort_values('month')
    
    # Create subplot with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces for each department
    departments = monthly_data['department'].unique()
    colors = px.colors.qualitative.Pastel
    
    for i, dept in enumerate(departments):
        dept_data = monthly_data[monthly_data['department'] == dept]
        
        fig.add_trace(
            go.Scatter(
                x=dept_data['month'], 
                y=dept_data['teaching_hours'],
                name=f'{dept} - Teaching',
                mode='lines+markers',
                line=dict(color=colors[i % len(colors)], width=2)
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=dept_data['month'], 
                y=dept_data['ai_usage_hours'],
                name=f'{dept} - AI Usage',
                mode='lines+markers',
                line=dict(color=colors[i % len(colors)], width=2, dash='dash')
            ),
            secondary_y=True
        )
    
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Teaching Hours", secondary_y=False)
    fig.update_yaxes(title_text="AI Usage Hours", secondary_y=True)
    
    fig.update_layout(
        title="Faculty Teaching Hours vs AI Usage Hours (Monthly Trend)",
        height=500,
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate statistics
    avg_teaching = plot_data['teaching_hours'].mean()
    avg_ai_usage = plot_data['ai_usage_hours'].mean()
    ai_ratio = (avg_ai_usage / avg_teaching) * 100 if avg_teaching > 0 else 0
    
    monthly_avg_teaching = plot_data.groupby('month')['teaching_hours'].mean()
    peak_teaching_month = monthly_avg_teaching.idxmax()
    
    # Create flashcards
    col1, col2 = st.columns(2)
    
    with col1:
        stats_content = f"""
        <b>Statistical Summary:</b><br>
        • Avg Teaching Hours: {avg_teaching:.2f}h<br>
        • Avg AI Usage Hours: {avg_ai_usage:.2f}h<br>
        • AI-to-Teaching Ratio: {ai_ratio:.1f}%<br>
        • Peak Teaching Month: {peak_teaching_month}<br>
        • Teaching Range: {plot_data['teaching_hours'].min():.1f} - {plot_data['teaching_hours'].max():.1f}h<br>
        • AI Usage Range: {plot_data['ai_usage_hours'].min():.1f} - {plot_data['ai_usage_hours'].max():.1f}h<br>
        • Workload Variance: {plot_data['teaching_hours'].std():.2f}
        """
        st.markdown(create_flashcard("Statistical Analysis", stats_content, "stats"), 
                    unsafe_allow_html=True)
    
    with col2:
        insights_content = f"""
        <b>Key Business Insights:</b><br>
        • <b>Time Investment:</b> Faculty spend {ai_ratio:.1f}% of teaching time on AI tools<br>
        • <b>Peak Period:</b> {peak_teaching_month} shows highest workload - plan support resources<br>
        • <b>Efficiency Gain:</b> AI tools potentially save {avg_teaching - avg_ai_usage:.1f}h per faculty<br>
        • <b>Resource Planning:</b> Consistent teaching hours with variable AI usage patterns<br>
        • <b>Training ROI:</b> Investment in AI tools yields measurable time optimization
        """
        st.markdown(create_flashcard("Business Insights", insights_content, "insights"), 
                    unsafe_allow_html=True)

def plot_faculty_performance_by_designation(data_dict, filters):
    """Create faculty performance comparison by designation with flashcards"""
    
    st.subheader("🎓 Faculty Performance by Designation")
    
    # Prepare data
    faculty = apply_filters(data_dict['faculty_master'], filters)
    faculty_metrics = data_dict['faculty_metrics'][
        data_dict['faculty_metrics']['faculty_id'].isin(faculty['faculty_id'])
    ]
    
    # Merge with master data
    plot_data = faculty_metrics.merge(faculty, on='faculty_id')
    
    # Aggregate
    agg_method = st.session_state.get('aggregation_method', 'mean')
    if agg_method == 'mean':
        agg_data = plot_data.groupby('designation').agg({
            'efficiency_score': 'mean',
            'student_avg_score': 'mean',
            'ai_dependency_ratio': 'mean'
        }).reset_index()
    elif agg_method == 'latest':
        latest_data = plot_data.sort_values('month').groupby('faculty_id').tail(1)
        agg_data = latest_data.groupby('designation').agg({
            'efficiency_score': 'mean',
            'student_avg_score': 'mean',
            'ai_dependency_ratio': 'mean'
        }).reset_index()
    else:
        agg_data = plot_data.groupby('designation').agg({
            'efficiency_score': 'mean',
            'student_avg_score': 'mean',
            'ai_dependency_ratio': 'mean'
        }).reset_index()
    
    # Create grouped bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Efficiency Score',
        x=agg_data['designation'],
        y=agg_data['efficiency_score'],
        marker_color='#ff7f0e',
        text=agg_data['efficiency_score'].round(1),
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        name='Student Avg Score',
        x=agg_data['designation'],
        y=agg_data['student_avg_score'],
        marker_color='#2ca02c',
        text=agg_data['student_avg_score'].round(1),
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Faculty Performance Metrics by Designation',
        xaxis_title='Designation',
        yaxis_title='Score',
        barmode='group',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate statistics
    best_designation = agg_data.loc[agg_data['efficiency_score'].idxmax(), 'designation']
    designation_counts = plot_data.groupby('designation')['faculty_id'].nunique()
    
    # Create flashcards
    col1, col2 = st.columns(2)
    
    with col1:
        stats_content = f"""
        <b>Statistical Summary:</b><br>
        • Best Efficiency: {best_designation} ({agg_data['efficiency_score'].max():.2f})<br>
        • Avg Efficiency Range: {agg_data['efficiency_score'].min():.2f} - {agg_data['efficiency_score'].max():.2f}<br>
        • Best Student Scores: {agg_data.loc[agg_data['student_avg_score'].idxmax(), 'designation']}<br>
        <b>Faculty Count by Designation:</b><br>
        """
        for desig, count in designation_counts.items():
            stats_content += f"• {desig}: {count} faculty<br>"
        
        st.markdown(create_flashcard("Statistical Analysis", stats_content, "stats"), 
                    unsafe_allow_html=True)
    
    with col2:
        insights_content = f"""
        <b>Key Business Insights:</b><br>
        • <b>Leadership Excellence:</b> {best_designation} shows highest efficiency - promote their methods<br>
        • <b>Experience Impact:</b> Senior designations demonstrate measurable performance advantages<br>
        • <b>Mentorship Program:</b> Pair junior faculty with {best_designation} for knowledge transfer<br>
        • <b>Career Development:</b> Clear performance progression validates promotion criteria<br>
        • <b>Hiring Strategy:</b> Current designation distribution aligns with performance needs
        """
        st.markdown(create_flashcard("Business Insights", insights_content, "insights"), 
                    unsafe_allow_html=True)

def plot_faculty_experience_analysis(data_dict, filters):
    """Create faculty experience vs performance analysis with flashcards"""
    
    st.subheader("📚 Faculty Experience vs Performance")
    
    # Prepare data
    faculty = apply_filters(data_dict['faculty_master'], filters)
    faculty_metrics = data_dict['faculty_metrics'][
        data_dict['faculty_metrics']['faculty_id'].isin(faculty['faculty_id'])
    ]
    
    # Merge with master data
    plot_data = faculty_metrics.merge(faculty, on='faculty_id')
    
    # Create experience bins
    plot_data['experience_category'] = pd.cut(
        plot_data['experience_years'], 
        bins=[0, 5, 10, 15, 25], 
        labels=['0-5 years', '6-10 years', '11-15 years', '15+ years']
    )
    
    # Aggregate
    agg_data = plot_data.groupby('experience_category').agg({
        'efficiency_score': 'mean',
        'student_avg_score': 'mean',
        'faculty_id': 'nunique'
    }).reset_index()
    agg_data.columns = ['experience_category', 'efficiency_score', 'student_avg_score', 'faculty_count']
    
    # Create line chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=agg_data['experience_category'],
        y=agg_data['efficiency_score'],
        mode='lines+markers',
        name='Efficiency Score',
        line=dict(color='#d62728', width=3),
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=agg_data['experience_category'],
        y=agg_data['student_avg_score'],
        mode='lines+markers',
        name='Student Avg Score',
        line=dict(color='#9467bd', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title='Faculty Performance by Experience Level',
        xaxis_title='Experience Category',
        yaxis_title='Score',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate statistics
    correlation = plot_data[['experience_years', 'efficiency_score']].corr().iloc[0, 1]
    
    # Create flashcards
    col1, col2 = st.columns(2)
    
    with col1:
        stats_content = f"""
        <b>Statistical Summary:</b><br>
        • Experience-Efficiency Correlation: {correlation:.3f}<br>
        • Most Experienced Group: 15+ years ({agg_data[agg_data['experience_category']=='15+ years']['faculty_count'].values[0] if '15+ years' in agg_data['experience_category'].values else 0} faculty)<br>
        • Junior Faculty: 0-5 years ({agg_data[agg_data['experience_category']=='0-5 years']['faculty_count'].values[0] if '0-5 years' in agg_data['experience_category'].values else 0} faculty)<br>
        • Avg Efficiency Improvement: {(agg_data['efficiency_score'].max() - agg_data['efficiency_score'].min()):.2f} points<br>
        • Total Faculty Analyzed: {plot_data['faculty_id'].nunique()}
        """
        st.markdown(create_flashcard("Statistical Analysis", stats_content, "stats"), 
                    unsafe_allow_html=True)
    
    with col2:
        insights_content = f"""
        <b>Key Business Insights:</b><br>
        • <b>Experience Dividend:</b> {correlation:.3f} correlation shows experience drives efficiency<br>
        • <b>Retention Priority:</b> Experienced faculty are productivity multipliers - focus on retention<br>
        • <b>Onboarding Gap:</b> Junior faculty need intensive support in first 5 years<br>
        • <b>Succession Planning:</b> Build knowledge transfer programs before senior retirements<br>
        • <b>Investment ROI:</b> Long-term faculty development yields measurable returns
        """
        st.markdown(create_flashcard("Business Insights", insights_content, "insights"), 
                    unsafe_allow_html=True)

def plot_comparative_ai_usage(data_dict, filters):
    """Create comparative AI usage analysis for students vs faculty with flashcards"""
    
    st.subheader("🔄 Comparative AI Usage: Students vs Faculty")
    
    # Prepare data
    students = apply_filters(data_dict['students_master'], filters)
    faculty = apply_filters(data_dict['faculty_master'], filters)
    
    student_metrics = data_dict['student_metrics'][
        data_dict['student_metrics']['student_id'].isin(students['student_id'])
    ]
    faculty_metrics = data_dict['faculty_metrics'][
        data_dict['faculty_metrics']['faculty_id'].isin(faculty['faculty_id'])
    ]
    
    # Merge with master data
    student_data = student_metrics.merge(students, on='student_id')
    faculty_data = faculty_metrics.merge(faculty, on='faculty_id')
    
    # Aggregate by department
    student_ai = student_data.groupby('department')['ai_usage_hours'].mean().reset_index()
    student_ai['type'] = 'Students'
    
    faculty_ai = faculty_data.groupby('department')['ai_usage_hours'].mean().reset_index()
    faculty_ai['type'] = 'Faculty'
    
    # Combine data
    combined_data = pd.concat([student_ai, faculty_ai])
    
    # Create grouped bar chart
    fig = px.bar(
        combined_data,
        x='department',
        y='ai_usage_hours',
        color='type',
        barmode='group',
        title='AI Usage Hours: Students vs Faculty by Department',
        labels={'ai_usage_hours': 'Average AI Usage Hours', 'department': 'Department'},
        color_discrete_map={'Students': '#3498db', 'Faculty': '#e74c3c'},
        template='plotly_white'
    )
    
    fig.update_layout(
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate statistics
    student_avg = student_data['ai_usage_hours'].mean()
    faculty_avg = faculty_data['ai_usage_hours'].mean()
    usage_gap = abs(student_avg - faculty_avg)
    
    # Create flashcards
    col1, col2 = st.columns(2)
    
    with col1:
        stats_content = f"""
        <b>Statistical Summary:</b><br>
        • Student Avg AI Usage: {student_avg:.2f}h<br>
        • Faculty Avg AI Usage: {faculty_avg:.2f}h<br>
        • Usage Gap: {usage_gap:.2f}h<br>
        • {'Faculty' if faculty_avg > student_avg else 'Students'} lead by {usage_gap:.2f}h<br>
        <b>Department Leaders:</b><br>
        • Student: {student_ai.loc[student_ai['ai_usage_hours'].idxmax(), 'department']}<br>
        • Faculty: {faculty_ai.loc[faculty_ai['ai_usage_hours'].idxmax(), 'department']}
        """
        st.markdown(create_flashcard("Statistical Analysis", stats_content, "stats"), 
                    unsafe_allow_html=True)
    
    with col2:
        insights_content = f"""
        <b>Key Business Insights:</b><br>
        • <b>Leadership Example:</b> Faculty {'setting strong precedent' if faculty_avg > student_avg else 'need to model AI adoption'}<br>
        • <b>Alignment Opportunity:</b> {usage_gap:.2f}h gap suggests training synchronization needed<br>
        • <b>Department Variance:</b> Inconsistent adoption across departments - share best practices<br>
        • <b>Resource Allocation:</b> Invest in AI infrastructure where gaps are widest<br>
        • <b>Culture Building:</b> Foster AI-first environment from faculty leadership
        """
        st.markdown(create_flashcard("Business Insights", insights_content, "insights"), 
                    unsafe_allow_html=True)

def plot_department_performance_scorecard(data_dict, filters):
    """Create comprehensive department performance scorecard with flashcards"""
    
    st.subheader("🏆 Department Performance Scorecard")
    
    # Prepare data
    students = apply_filters(data_dict['students_master'], filters)
    faculty = apply_filters(data_dict['faculty_master'], filters)
    
    student_metrics = data_dict['student_metrics'][
        data_dict['student_metrics']['student_id'].isin(students['student_id'])
    ]
    faculty_metrics = data_dict['faculty_metrics'][
        data_dict['faculty_metrics']['faculty_id'].isin(faculty['faculty_id'])
    ]
    
    # Merge with master data
    student_data = student_metrics.merge(students, on='student_id')
    faculty_data = faculty_metrics.merge(faculty, on='faculty_id')
    
    # Calculate department metrics
    dept_metrics = []
    
    for dept in student_data['department'].unique():
        dept_students = student_data[student_data['department'] == dept]
        dept_faculty = faculty_data[faculty_data['department'] == dept]
        
        metrics = {
            'Department': dept,
            'Avg Student Score': dept_students['avg_score'].mean(),
            'Avg Attendance': dept_students['attendance_pct'].mean(),
            'Avg Engagement': dept_students['engagement_score'].mean(),
            'Faculty Efficiency': dept_faculty['efficiency_score'].mean(),
            'Student Count': dept_students['student_id'].nunique(),
            'Faculty Count': dept_faculty['faculty_id'].nunique()
        }
        dept_metrics.append(metrics)
    
    scorecard_df = pd.DataFrame(dept_metrics)
    
    # Normalize scores for radar chart (0-100 scale)
    radar_data = scorecard_df.copy()
    for col in ['Avg Student Score', 'Avg Attendance', 'Avg Engagement', 'Faculty Efficiency']:
        radar_data[col] = (radar_data[col] / radar_data[col].max()) * 100
    
    # Create radar chart
    fig = go.Figure()
    
    categories = ['Student Score', 'Attendance', 'Engagement', 'Faculty Efficiency']
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, row in radar_data.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['Avg Student Score'], row['Avg Attendance'], 
               row['Avg Engagement'], row['Faculty Efficiency']],
            theta=categories,
            fill='toself',
            name=row['Department'],
            line_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=True,
        title='Department Performance Radar Chart (Normalized Scores)',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display scorecard table
    st.subheader("Detailed Scorecard")
    
    # Format the scorecard
    scorecard_display = scorecard_df.copy()
    scorecard_display['Avg Student Score'] = scorecard_display['Avg Student Score'].round(2)
    scorecard_display['Avg Attendance'] = scorecard_display['Avg Attendance'].round(2)
    scorecard_display['Avg Engagement'] = scorecard_display['Avg Engagement'].round(2)
    scorecard_display['Faculty Efficiency'] = scorecard_display['Faculty Efficiency'].round(2)
    
    st.dataframe(scorecard_display, use_container_width=True, hide_index=True)
    
    # Calculate statistics
    best_overall = scorecard_df.loc[
        (scorecard_df['Avg Student Score'] + scorecard_df['Faculty Efficiency']).idxmax(), 
        'Department'
    ]
    
    # Create flashcards
    col1, col2 = st.columns(2)
    
    with col1:
        stats_content = f"""
        <b>Statistical Summary:</b><br>
        • Best Overall: {best_overall}<br>
        • Highest Student Score: {scorecard_df.loc[scorecard_df['Avg Student Score'].idxmax(), 'Department']} ({scorecard_df['Avg Student Score'].max():.2f})<br>
        • Highest Attendance: {scorecard_df.loc[scorecard_df['Avg Attendance'].idxmax(), 'Department']} ({scorecard_df['Avg Attendance'].max():.2f}%)<br>
        • Highest Engagement: {scorecard_df.loc[scorecard_df['Avg Engagement'].idxmax(), 'Department']} ({scorecard_df['Avg Engagement'].max():.2f})<br>
        • Most Efficient Faculty: {scorecard_df.loc[scorecard_df['Faculty Efficiency'].idxmax(), 'Department']} ({scorecard_df['Faculty Efficiency'].max():.2f})
        """
        st.markdown(create_flashcard("Statistical Analysis", stats_content, "stats"), 
                    unsafe_allow_html=True)
    
    with col2:
        insights_content = f"""
        <b>Key Business Insights:</b><br>
        • <b>Excellence Model:</b> {best_overall} demonstrates balanced high performance - institution benchmark<br>
        • <b>Resource Reallocation:</b> Direct support to underperforming departments<br>
        • <b>Best Practice Sharing:</b> Create cross-department learning forums<br>
        • <b>Strategic Investment:</b> Scale successful {best_overall} practices institution-wide<br>
        • <b>Competitive Advantage:</b> Balanced scorecard reveals holistic departmental health
        """
        st.markdown(create_flashcard("Business Insights", insights_content, "insights"), 
                    unsafe_allow_html=True)

# ================================
# MAIN APPLICATION
# ================================

def main():
    """Main application function"""
    
    # Load data
    data_dict = load_data()
    
    if data_dict is None:
        st.error("Failed to load data. Please check the file paths.")
        return
    
    # Display main title
    st.markdown('<h1 class="main-title">📊 Education Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # ================================
    # SIDEBAR - GLOBAL FILTERS
    # ================================
    
    st.sidebar.title("🎛️ Global Filters")
    
    # Department filter
    all_departments = sorted(data_dict['students_master']['department'].unique().tolist())
    selected_departments = st.sidebar.multiselect(
        "Select Departments",
        options=all_departments,
        default=all_departments
    )
    
    # Month filter
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    selected_months = st.sidebar.multiselect(
        "Select Months",
        options=month_order,
        default=month_order
    )
    
    # Aggregation method
    aggregation_method = st.sidebar.selectbox(
        "Data Aggregation Method",
        options=['mean', 'latest', 'sum'],
        index=0,
        help="Choose how to aggregate monthly data"
    )
    
    # Store in session state
    st.session_state['aggregation_method'] = aggregation_method
    
    # Apply global filters
    global_filters = {
        'department': selected_departments if selected_departments else all_departments
    }
    
    # Filter metrics by selected months
    if selected_months:
        data_dict['student_metrics'] = data_dict['student_metrics'][
            data_dict['student_metrics']['month'].isin(selected_months)
        ]
        data_dict['faculty_metrics'] = data_dict['faculty_metrics'][
            data_dict['faculty_metrics']['month'].isin(selected_months)
        ]
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Active Filters:**\n- Departments: {len(selected_departments)}\n- Months: {len(selected_months)}")
    
    # ================================
    # PAGE NAVIGATION
    # ================================
    
    page = st.sidebar.radio(
        "Navigate to:",
        ["📌 Executive Overview", "🎓 Student Analytics", "👨‍🏫 Faculty Analytics", "🔀 Comparative Analysis"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔔 Anomaly Alerts")
    
    # Calculate anomalies
    students_filtered = apply_filters(data_dict['students_master'], global_filters)
    faculty_filtered = apply_filters(data_dict['faculty_master'], global_filters)
    
    student_metrics_filtered = data_dict['student_metrics'][
        data_dict['student_metrics']['student_id'].isin(students_filtered['student_id'])
    ]
    faculty_metrics_filtered = data_dict['faculty_metrics'][
        data_dict['faculty_metrics']['faculty_id'].isin(faculty_filtered['faculty_id'])
    ]
    
    low_attendance_count = len(student_metrics_filtered[student_metrics_filtered['attendance_pct'] < 70])
    low_score_count = len(student_metrics_filtered[student_metrics_filtered['avg_score'] < 40])
    high_ai_dependency_count = len(faculty_metrics_filtered[faculty_metrics_filtered['ai_dependency_ratio'] > 1.0])
    
    if low_attendance_count > 0:
        st.sidebar.markdown(f'<span class="anomaly-badge">🚨 {low_attendance_count} Low Attendance Cases</span>', 
                            unsafe_allow_html=True)
    
    if low_score_count > 0:
        st.sidebar.markdown(f'<span class="anomaly-badge">⚠️ {low_score_count} Low Score Cases</span>', 
                            unsafe_allow_html=True)
    
    if high_ai_dependency_count > 0:
        st.sidebar.markdown(f'<span class="warning-badge">🤖 {high_ai_dependency_count} High AI Dependency</span>', 
                            unsafe_allow_html=True)
    
    # ================================
    # PAGE CONTENT
    # ================================
    
    if page == "📌 Executive Overview":
        st.header("Executive Overview")
        
        # KPI Cards
        create_kpi_cards(data_dict, global_filters)
        
        st.markdown("---")
        
        # Department Performance Overview
        plot_department_performance_scorecard(data_dict, global_filters)
        
        st.markdown("---")
        
        # Monthly Trends
        col1, col2 = st.columns(2)
        
        with col1:
            plot_student_performance_trends(data_dict, global_filters)
        
        with col2:
            plot_faculty_teaching_hours_analysis(data_dict, global_filters)
    
    elif page == "🎓 Student Analytics":
        st.header("Student Analytics")
        
        # Local filters for student page
        col1, col2, col3 = st.columns(3)
        
        with col1:
            degree_levels = st.multiselect(
                "Degree Level",
                options=data_dict['students_master']['degree_level'].unique().tolist(),
                default=data_dict['students_master']['degree_level'].unique().tolist()
            )
        
        with col2:
            years_of_study = st.multiselect(
                "Year of Study",
                options=sorted(data_dict['students_master']['year_of_study'].unique().tolist()),
                default=sorted(data_dict['students_master']['year_of_study'].unique().tolist())
            )
        
        with col3:
            score_range = st.slider(
                "Score Range",
                min_value=0,
                max_value=100,
                value=(0, 100)
            )
        
        # Apply local filters
        local_filters = global_filters.copy()
        if degree_levels:
            local_filters['degree_level'] = degree_levels
        if years_of_study:
            local_filters['year_of_study'] = years_of_study
        
        # Filter by score range
        students_filtered = apply_filters(data_dict['students_master'], local_filters)
        student_metrics_filtered = data_dict['student_metrics'][
            data_dict['student_metrics']['student_id'].isin(students_filtered['student_id'])
        ]
        student_metrics_filtered = student_metrics_filtered[
            (student_metrics_filtered['avg_score'] >= score_range[0]) &
            (student_metrics_filtered['avg_score'] <= score_range[1])
        ]
        
        # Update data_dict for this page
        page_data_dict = data_dict.copy()
        page_data_dict['students_master'] = students_filtered
        page_data_dict['student_metrics'] = student_metrics_filtered
        
        st.markdown("---")
        
        # Student visualizations
        plot_student_attendance_vs_performance(page_data_dict, {})
        
        st.markdown("---")
        
        plot_student_ai_usage_distribution(page_data_dict, {})
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            plot_student_performance_trends(page_data_dict, {})
        
        with col2:
            plot_student_engagement_analysis(page_data_dict, {})
    
    elif page == "👨‍🏫 Faculty Analytics":
        st.header("Faculty Analytics")
        
        # Local filters for faculty page
        col1, col2 = st.columns(2)
        
        with col1:
            designations = st.multiselect(
                "Designation",
                options=data_dict['faculty_master']['designation'].unique().tolist(),
                default=data_dict['faculty_master']['designation'].unique().tolist()
            )
        
        with col2:
            experience_range = st.slider(
                "Experience (Years)",
                min_value=int(data_dict['faculty_master']['experience_years'].min()),
                max_value=int(data_dict['faculty_master']['experience_years'].max()),
                value=(int(data_dict['faculty_master']['experience_years'].min()),
                       int(data_dict['faculty_master']['experience_years'].max()))
            )
        
        # Apply local filters
        local_filters = global_filters.copy()
        if designations:
            local_filters['designation'] = designations
        
        # Filter by experience range
        faculty_filtered = apply_filters(data_dict['faculty_master'], local_filters)
        faculty_filtered = faculty_filtered[
            (faculty_filtered['experience_years'] >= experience_range[0]) &
            (faculty_filtered['experience_years'] <= experience_range[1])
        ]
        faculty_metrics_filtered = data_dict['faculty_metrics'][
            data_dict['faculty_metrics']['faculty_id'].isin(faculty_filtered['faculty_id'])
        ]
        
        # Update data_dict for this page
        page_data_dict = data_dict.copy()
        page_data_dict['faculty_master'] = faculty_filtered
        page_data_dict['faculty_metrics'] = faculty_metrics_filtered
        
        st.markdown("---")
        
        # Faculty visualizations
        plot_faculty_ai_efficiency(page_data_dict, {})
        
        st.markdown("---")
        
        plot_faculty_teaching_hours_analysis(page_data_dict, {})
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            plot_faculty_performance_by_designation(page_data_dict, {})
        
        with col2:
            plot_faculty_experience_analysis(page_data_dict, {})
    
    elif page == "🔀 Comparative Analysis":
        st.header("Comparative Analysis")
        
        st.markdown("Compare student and faculty metrics across departments")
        
        st.markdown("---")
        
        plot_comparative_ai_usage(data_dict, global_filters)
        
        st.markdown("---")
        
        # Correlation Analysis
        st.subheader("📊 Correlation Matrix: Key Metrics")
        
        # Prepare correlation data
        students = apply_filters(data_dict['students_master'], global_filters)
        faculty = apply_filters(data_dict['faculty_master'], global_filters)
        
        student_metrics = data_dict['student_metrics'][
            data_dict['student_metrics']['student_id'].isin(students['student_id'])
        ]
        faculty_metrics = data_dict['faculty_metrics'][
            data_dict['faculty_metrics']['faculty_id'].isin(faculty['faculty_id'])
        ]
        
        # Calculate correlations
        student_corr = student_metrics[['attendance_pct', 'avg_score', 'ai_usage_hours', 
                                        'performance_delta', 'engagement_score']].corr()
        
        # Create heatmap
        fig = px.imshow(
            student_corr,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title='Student Metrics Correlation Matrix',
            labels=dict(color="Correlation")
        )
        
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create flashcards for correlation analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Find strongest correlations
            corr_values = student_corr.values
            np.fill_diagonal(corr_values, 0)  # Remove diagonal
            max_corr_idx = np.unravel_index(np.argmax(np.abs(corr_values)), corr_values.shape)
            strongest_pair = (student_corr.index[max_corr_idx[0]], student_corr.columns[max_corr_idx[1]])
            strongest_value = corr_values[max_corr_idx]
            
            stats_content = f"""
            <b>Correlation Insights:</b><br>
            • Strongest Correlation: {strongest_pair[0]} ↔ {strongest_pair[1]} ({strongest_value:.3f})<br>
            • Attendance ↔ Score: {student_corr.loc['attendance_pct', 'avg_score']:.3f}<br>
            • AI Usage ↔ Performance: {student_corr.loc['ai_usage_hours', 'performance_delta']:.3f}<br>
            • Engagement ↔ Score: {student_corr.loc['engagement_score', 'avg_score']:.3f}<br>
            • AI Usage ↔ Score: {student_corr.loc['ai_usage_hours', 'avg_score']:.3f}
            """
            st.markdown(create_flashcard("Statistical Analysis", stats_content, "stats"), 
                        unsafe_allow_html=True)
        
        with col2:
            insights_content = f"""
            <b>Key Business Insights:</b><br>
            • <b>Primary Driver:</b> {strongest_pair[0]} most strongly influences {strongest_pair[1]}<br>
            • <b>Intervention Focus:</b> Target metrics with weak correlations for potential improvement<br>
            • <b>Resource Optimization:</b> Invest in factors showing strong correlation with outcomes<br>
            • <b>Holistic Approach:</b> Multiple metrics impact performance - avoid single-factor focus<br>
            • <b>Data-Driven Strategy:</b> Use correlation patterns to prioritize institutional initiatives
            """
            st.markdown(create_flashcard("Business Insights", insights_content, "insights"), 
                        unsafe_allow_html=True)
        
        st.markdown("---")
        
        plot_department_performance_scorecard(data_dict, global_filters)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>Educational Institution Analytics Dashboard | Built with Streamlit & Plotly</p>
            <p>Data-driven insights for academic excellence</p>
        </div>
    """, unsafe_allow_html=True)

# ================================
# RUN APPLICATION
# ================================

if __name__ == "__main__":
    main()
