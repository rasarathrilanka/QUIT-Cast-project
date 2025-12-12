import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, Cell
} from 'recharts';
import './App.css';

const API_BASE_URL = 'http://localhost:5000/api';

// Color scheme
const COLORS = {
  veryHighRisk: '#ef4444',
  highRisk: '#f97316',
  mediumRisk: '#f59e0b',
  lowRisk: '#10b981',
  primary: '#3b82f6',
  secondary: '#8b5cf6',
  danger: '#ef4444',
  warning: '#f59e0b',
  success: '#10b981'
};


function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [loading, setLoading] = useState(false);
  const [roles, setRoles] = useState([]);
  const [maritalStatuses, setMaritalStatuses] = useState([]);
  const [employeeType, setEmployeeType] = useState('existing');
  const [csvFile, setCsvFile] = useState(null);
  const [csvResults, setCsvResults] = useState(null);
  const [selectedYear, setSelectedYear] = useState(null);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadedAnalytics, setUploadedAnalytics] = useState(null);
  const [employeeId, setEmployeeId] = useState('');
  const [employeeIdResult, setEmployeeIdResult] = useState(null);
  const [selectedDepartment, setSelectedDepartment] = useState('Engineering');
  const [departmentPrediction, setDepartmentPrediction] = useState(null);
  const [deptSelectedYear, setDeptSelectedYear] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [modalData, setModalData] = useState({ title: '', employees: [] });
  const [factorModal, setFactorModal] = useState(false);
  const [factorData, setFactorData] = useState(null);

const [showAdvancedFactors, setShowAdvancedFactors] = useState(false);

// Insights state
const [insightsTab, setInsightsTab] = useState('optimizer');
const [selectedEmployeeForStrategy, setSelectedEmployeeForStrategy] = useState(null);
const [retentionStrategies, setRetentionStrategies] = useState(null);
const [clusteringData, setClusteringData] = useState(null);
const [selectedCluster, setSelectedCluster] = useState(null);
const [loadingInsights, setLoadingInsights] = useState(false);

const [existingEmployeeFactors, setExistingEmployeeFactors] = useState(null);
const [singleEmployeeFactors, setSingleEmployeeFactors] = useState(null);
  const [singleEmployee, setSingleEmployee] = useState({
    employee_name: '',
    age: 28,
    time_at_current_role: 1.5,
    marital_status: 'Single',
    role: 'Software Engineer',
    work_experience: 3.0,
    wfh_available: 1
  });
  const [singlePrediction, setSinglePrediction] = useState(null);

  useEffect(() => {
    fetchOptions();
    fetchUploadedAnalytics();
  }, []);

  const fetchOptions = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/options/all`);
      const data = await response.json();
      setRoles(data.roles);
      setMaritalStatuses(data.marital_statuses);
    } catch (error) {
      console.error('Error fetching options:', error);
    }
  };

const handleSinglePrediction = async (e) => {
  e.preventDefault();
  setLoading(true);

  try {
    // Smart defaults for missing contextual factors
    const enhancedEmployee = {
      ...singleEmployee,
      // Calculate salary satisfaction based on role and experience
      salary_satisfaction: calculateSalarySatisfaction(singleEmployee),
      // COVID impact based on work experience (joined during COVID?)
      covid_impact_score: calculateCovidImpact(singleEmployee.work_experience),
      // Economic crisis impact (2022 Sri Lanka crisis)
      economic_crisis_impact: calculateEconomicImpact(singleEmployee.work_experience),
      // Political stability concern (moderate default)
      political_stability_concern: 5.5
    };

    const response = await fetch(`${API_BASE_URL}/predict/single`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(enhancedEmployee)
    });

    const data = await response.json();
    setSinglePrediction(data);

    // Fetch factor analysis if we have employee data
    if (data.attrition_probability) {
      fetchSingleEmployeeFactors(enhancedEmployee);
    }
  } catch (error) {
    console.error('Error:', error);
    alert('Error making prediction');
  } finally {
    setLoading(false);
  }
};

const calculateSalarySatisfaction = (employee) => {
  // Junior roles: likely underpaid
  const juniorRoles = ['Trainee Developer', 'Junior Developer'];
  // Senior roles: likely well paid
  const seniorRoles = ['Tech Lead', 'Engineering Manager', 'Architect', 'Director'];

  if (juniorRoles.includes(employee.role)) {
    return -0.3; // Slightly underpaid
  } else if (seniorRoles.includes(employee.role)) {
    return employee.work_experience > 8 ? 0.2 : -0.1;
  } else {
    // Mid-level: depends on experience vs time in role
    if (employee.time_at_current_role > 4) {
      return -0.4; // Stuck in role, likely underpaid
    }
    return -0.15; // Slightly underpaid (average)
  }
};

const calculateCovidImpact = (workExperience) => {
  // If they have 2-4 years experience, they joined during COVID (2020-2022)
  if (workExperience >= 2 && workExperience <= 4) {
    return 8.5; // High impact
  } else if (workExperience > 4 && workExperience <= 6) {
    return 5.0; // Medium impact
  }
  return 2.0; // Low impact
};

const calculateEconomicImpact = (workExperience) => {
  // 2022 Sri Lanka economic crisis
  // If they have 2+ years, they were present during crisis
  if (workExperience >= 2) {
    return 8.0; // High impact
  } else if (workExperience >= 1) {
    return 4.5; // Medium impact
  }
  return 1.5; // Minimal impact
};

const fetchSingleEmployeeFactors = async (employeeData) => {
  try {
    // Create a mock endpoint call with the employee data
    const response = await fetch(`${API_BASE_URL}/predict/single/factors`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(employeeData)
    });

    const data = await response.json();
    if (response.ok) {
      setSingleEmployeeFactors(data);
    }
  } catch (error) {
    console.error('Error fetching factors:', error);
  }
};

  const handleCsvUpload = async (e) => {
    e.preventDefault();

    if (!csvFile) {
      alert('Please select a CSV file');
      return;
    }

    setUploadLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', csvFile);

      const response = await fetch(`${API_BASE_URL}/upload/csv`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (response.ok) {
        setCsvResults(data);
        setSelectedYear(data.years[0].year);
        fetchUploadedAnalytics();
        alert('CSV uploaded successfully! View analytics on Home page.');
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error:', error);
      alert('Error uploading CSV');
    } finally {
      setUploadLoading(false);
    }
  };

  const fetchUploadedAnalytics = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/analytics/uploaded`);
      const data = await response.json();
      if (response.ok) {
        setUploadedAnalytics(data);
      }
    } catch (error) {
      console.error('Error fetching analytics:', error);
    }
  };

  const showHighRiskEmployees = () => {
    if (!uploadedAnalytics) {
      alert('Please upload CSV data first');
      return;
    }

    const highRisk = uploadedAnalytics.high_risk_employees || [];

    if (highRisk.length === 0) {
      alert('No high risk employees found');
      return;
    }

    setModalData({
      title: `High Risk Employees (${highRisk.length})`,
      employees: highRisk
    });
    setShowModal(true);
  };

  const showMediumRiskEmployees = () => {
    if (!uploadedAnalytics) {
      alert('Please upload CSV data first');
      return;
    }

    const mediumRisk = uploadedAnalytics.medium_risk_employees || [];

    if (mediumRisk.length === 0) {
      alert('No medium risk employees found');
      return;
    }

    setModalData({
      title: `Medium Risk Employees (${mediumRisk.length})`,
      employees: mediumRisk
    });
    setShowModal(true);
  };

  const showQuarterLeavers = (quarterData) => {
    if (!quarterData.employees) {
      alert('Employee data not available for this quarter');
      return;
    }

    const leavers = quarterData.employees
      .filter(emp => emp.attrition_probability >= 50)
      .sort((a, b) => b.attrition_probability - a.attrition_probability);

    if (leavers.length === 0) {
      alert('No expected leavers for this quarter');
      return;
    }

    setModalData({
      title: `Expected Leavers - ${quarterData.quarter_label} (${leavers.length})`,
      employees: leavers
    });
    setShowModal(true);
  };

  const showEmployeeFactors = async (employeeId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/predict/employee/${employeeId}/factors`);
      const data = await response.json();

      if (response.ok) {
        setFactorData(data);
        setFactorModal(true);
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error:', error);
      alert('Error fetching factor analysis');
    }
  };

const fetchRetentionStrategies = async (employeeId) => {
  setLoadingInsights(true);
  try {
    const response = await fetch(`${API_BASE_URL}/insights/retention-strategies/${employeeId}`);
    const data = await response.json();

    if (response.ok) {
      setRetentionStrategies(data);
    } else {
      alert(`Error: ${data.error}`);
    }
  } catch (error) {
    console.error('Error:', error);
    alert('Error fetching retention strategies');
  } finally {
    setLoadingInsights(false);
  }
};

const fetchClusteringData = async () => {
  setLoadingInsights(true);
  try {
    const response = await fetch(`${API_BASE_URL}/insights/clustering`);
    const data = await response.json();

    if (response.ok) {
      setClusteringData(data);
    } else {
      alert(`Error: ${data.error}`);
    }
  } catch (error) {
    console.error('Error:', error);
    alert('Error fetching clustering data');
  } finally {
    setLoadingInsights(false);
  }
};

const handleSelectEmployeeForStrategy = (employeeId) => {
  setSelectedEmployeeForStrategy(employeeId);
  fetchRetentionStrategies(employeeId);
};

const handleEmployeeIdSearch = async (e) => {
  e.preventDefault();

  if (!employeeId) {
    alert('Please enter an Employee ID');
    return;
  }

  setLoading(true);

  try {
    const response = await fetch(`${API_BASE_URL}/predict/employee/${employeeId}`);
    const data = await response.json();

    if (response.ok) {
      setEmployeeIdResult(data);

      // Fetch factor analysis for existing employee
      const factorsResponse = await fetch(`${API_BASE_URL}/predict/employee/${employeeId}/factors`);
      const factorsData = await factorsResponse.json();

      if (factorsResponse.ok) {
        setExistingEmployeeFactors(factorsData);
      }
    } else {
      alert(`Error: ${data.error}`);
      setEmployeeIdResult(null);
      setExistingEmployeeFactors(null);
    }
  } catch (error) {
    console.error('Error:', error);
    alert('Error fetching employee prediction');
  } finally {
    setLoading(false);
  }
};

  const handleDepartmentPrediction = async (dept) => {
    setSelectedDepartment(dept);
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/predict/department/${dept}`);
      const data = await response.json();

      if (response.ok) {
        setDepartmentPrediction(data);
        setDeptSelectedYear(data.years[0].year);
      } else {
        alert(`Error: ${data.error}`);
        setDepartmentPrediction(null);
      }
    } catch (error) {
      console.error('Error:', error);
      alert('Error fetching department prediction');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadSample = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/download/sample-csv`);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'sample_employees.csv';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error:', error);
      alert('Error downloading sample CSV');
    }
  };

  // Render functions continue on next file...
  const renderHome = () => (
    <div className="home-container">
      {!uploadedAnalytics && (
        <div className="upload-hero">
          <div className="upload-hero-content">
            <h1>📊 Employee Attrition Prediction System</h1>
            <p className="upload-subtitle">Upload your employee data to get AI-powered attrition insights</p>
            <div className="upload-box">
              <div className="upload-icon">📤</div>
              <h3>Upload Employee CSV</h3>
              <p>Drag and drop or click to select your CSV file</p>
              <input type="file" accept=".csv" onChange={(e) => setCsvFile(e.target.files[0])} id="csv-upload-input" style={{display: 'none'}} />
              <label htmlFor="csv-upload-input" className="btn-upload-large">
                {csvFile ? `📄 ${csvFile.name}` : '📁 Choose File'}
              </label>
              {csvFile && (
                <button onClick={handleCsvUpload} className="btn-process-large" disabled={uploadLoading}>
                  {uploadLoading ? '⏳ Processing...' : '✨ Analyze Data'}
                </button>
              )}
              <div className="upload-footer">
                <button onClick={handleDownloadSample} className="btn-link">📥 Download Sample CSV</button>
              </div>
            </div>
          </div>
        </div>
      )}

      {uploadedAnalytics && (
        <>
          <div className="analytics-header">
            <div>
              <h1>Dashboard Overview</h1>
              <p className="dashboard-subtitle">AI-Powered Workforce Analytics</p>
            </div>
            <div className="header-upload">
              <input type="file" accept=".csv" onChange={(e) => setCsvFile(e.target.files[0])} id="header-upload-input" style={{display: 'none'}} />
              <label htmlFor="header-upload-input" className="btn-upload-header">
                📤 {csvFile ? 'File Selected' : 'Upload New Data'}
              </label>
              {csvFile && (
                <button onClick={handleCsvUpload} className="btn-process-header" disabled={uploadLoading}>
                  {uploadLoading ? '⏳' : '✓'}
                </button>
              )}
            </div>
          </div>

          <div className="metrics-row">
            <div className="metric-box primary">
              <h3>Overall Attrition Rate</h3>
              <div className="metric-big-value">{uploadedAnalytics.overall_attrition_rate}%</div>
              <p className="metric-desc">Average across all employees</p>
            </div>
            <div className="metric-box danger clickable" onClick={() => showHighRiskEmployees()}>
              <h3>High Risk Employees</h3>
              <div className="metric-big-value">{uploadedAnalytics.high_risk_count}</div>
              <p className="metric-desc">👆 Click to view list</p>
            </div>
            <div className="metric-box warning clickable" onClick={() => showMediumRiskEmployees()}>
              <h3>Medium Risk</h3>
              <div className="metric-big-value">{uploadedAnalytics.medium_risk_count}</div>
              <p className="metric-desc">👆 Click to view list</p>
            </div>
            <div className="metric-box success">
              <h3>Total Employees</h3>
              <div className="metric-big-value">{uploadedAnalytics.total_employees}</div>
              <p className="metric-desc">Uploaded data</p>
            </div>
          </div>

          <div className="section">
            <h2 className="section-title">🚨 Top 10 High Risk Employees</h2>
            <div className="risk-grid">
              {uploadedAnalytics.top_risk.slice(0, 10).map((emp, idx) => (
                <div key={idx} className="risk-card" onClick={() => showEmployeeFactors(emp.employee_id)} style={{cursor: 'pointer'}}>
                  <div className="risk-rank">#{idx + 1}</div>
                  <div className="risk-card-content">
                    <strong className="employee-id">{emp.employee_id}</strong>
                    <span className="employee-role">{emp.role}</span>
                    <span className="employee-details">Age: {emp.age} • {emp.work_experience}yr exp</span>
                  </div>
                  <div className="risk-percentage" style={{backgroundColor: emp.attrition_probability >= 50 ? COLORS.veryHighRisk : COLORS.highRisk}}>
                    {emp.attrition_probability.toFixed(1)}%
                  </div>
                </div>
              ))}
            </div>
          </div>

          {csvResults && (
            <>
              <div className="section">
                <h2 className="section-title">📈 5-Year Attrition Trend</h2>
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height={350}>
                    <LineChart data={csvResults.years}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="year" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="avg_attrition_probability" stroke={COLORS.primary} strokeWidth={3} name="Attrition Probability (%)" dot={{ r: 6 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="section">
                <h2 className="section-title">📅 Quarterly Attrition Forecast</h2>
                <div className="year-selector-large">
                  {csvResults.years.map(yearData => (
                    <button key={yearData.year} className={`year-btn-large ${selectedYear === yearData.year ? 'active' : ''}`} onClick={() => setSelectedYear(yearData.year)}>
                      <span className="year-label-large">{yearData.year}</span>
                      <span className="year-value-large">{yearData.avg_attrition_probability}%</span>
                    </button>
                  ))}
                </div>
                {selectedYear && (
                  <div className="quarters-display">
                    {csvResults.all_predictions.filter(pred => pred.year === selectedYear).map(quarter => (
                      <div key={`${quarter.year}-${quarter.quarter}`} className="quarter-box">
                        <h4 className="quarter-title">{quarter.quarter_label}</h4>
                        <div className="quarter-value">{quarter.avg_attrition_probability}%</div>
                        <div className="quarter-divider"></div>
                        <p className="quarter-leavers-link" onClick={(e) => { e.stopPropagation(); showQuarterLeavers(quarter); }}>
                          👆 Expected Leavers: {quarter.expected_leavers}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}

          <div className="section">
            <h2 className="section-title">🏢 Attrition by Department</h2>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={Object.entries(uploadedAnalytics.by_department).map(([dept, data]) => ({department: dept, attrition: data.mean, count: data.count}))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="department" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="attrition" fill={COLORS.primary} name="Attrition %" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}
    </div>
  );

  const renderPrediction = () => (
    <div className="prediction-container">
      <div className="tabs">
        <button className={activeTab === 'single' ? 'active' : ''} onClick={() => setActiveTab('single')}>Single Employee</button>
        <button className={activeTab === 'department' ? 'active' : ''} onClick={() => setActiveTab('department')}>Department/Team</button>
      </div>

      {activeTab === 'single' && (
        <div className="single-prediction">
          <h2>Single Employee Prediction</h2>
          <div className="employee-type-tabs">
            <button className={employeeType === 'existing' ? 'active' : ''} onClick={() => setEmployeeType('existing')}>Existing Employee</button>
            <button className={employeeType === 'new' ? 'active' : ''} onClick={() => setEmployeeType('new')}>New Employee / Hire</button>
          </div>

          {employeeType === 'existing' && (
            <div className="existing-employee-section">
              <h3>Search by Employee ID</h3>
              <form onSubmit={handleEmployeeIdSearch}>
                <div className="search-group">
                  <input type="text" value={employeeId} onChange={(e) => setEmployeeId(e.target.value)} placeholder="Enter Employee ID (e.g., EMP0001)" className="search-input" />
                  <button type="submit" className="btn-primary" disabled={loading}>{loading ? 'Searching...' : 'Search'}</button>
                </div>
              </form>
                {employeeIdResult && (
                  <div className="prediction-result-enhanced">
                    <h3>Employee: {employeeIdResult.employee_data.employee_id}</h3>

                    <div className="result-layout">
                      {/* Left side - Big percentage */}
                      <div className="result-main">
                        <div className="result-primary-card">
                          <div className="employee-info-header">
                            <p><strong>Role:</strong> {employeeIdResult.employee_data.role}</p>
                            <p><strong>Age:</strong> {employeeIdResult.employee_data.age}</p>
                            <p><strong>Experience:</strong> {employeeIdResult.employee_data.work_experience} years</p>
                          </div>

                          <div className="divider"></div>

                          <h4>Attrition Probability</h4>
                          <div
                            className="probability-big"
                            style={{color: employeeIdResult.risk_color}}
                          >
                            {employeeIdResult.attrition_probability}%
                          </div>
                          <div className="risk-badge-large" style={{
                            backgroundColor: employeeIdResult.risk_color
                          }}>
                            {employeeIdResult.risk_icon} {employeeIdResult.risk_level}
                          </div>
                        </div>
                      </div>

                      {/* Right side - Factors */}
                      <div className="result-factors">
                        {existingEmployeeFactors ? (
                          <>
                            <div className="factors-chart-small">
                              <h4>Contributing Factors</h4>
                              <ResponsiveContainer width="100%" height={250}>
                                <BarChart
                                  data={existingEmployeeFactors.factors}
                                  layout="vertical"
                                  margin={{ left: 100, right: 20 }}
                                >
                                  <CartesianGrid strokeDasharray="3 3" />
                                  <XAxis type="number" domain={[0, 100]} />
                                  <YAxis type="category" dataKey="name" width={90} />
                                  <Tooltip />
                                  <Bar dataKey="value" fill="#3b82f6" radius={[0, 8, 8, 0]}>
                                    {existingEmployeeFactors.factors.map((entry, index) => (
                                      <Cell key={`cell-${index}`} fill={entry.color} />
                                    ))}
                                  </Bar>
                                </BarChart>
                              </ResponsiveContainer>
                            </div>

                            <div className="factors-list-small">
                              {existingEmployeeFactors.factors.map((factor, idx) => (
                                <div key={idx} className="factor-item-small">
                                  <div className="factor-header-small">
                                    <span
                                      className="factor-dot"
                                      style={{backgroundColor: factor.color}}
                                    ></span>
                                    <span className="factor-name">{factor.name}</span>
                                    <span className="factor-value">{factor.value.toFixed(1)}%</span>
                                  </div>
                                </div>
                              ))}
                            </div>

                            <div className="context-summary">
                              <h4>Contextual Factors</h4>
                              <div className="context-mini-grid">
                                <div className="context-mini">
                                  <span className="context-icon">💰</span>
                                  <span className="context-text">
                                    {existingEmployeeFactors.employee_data.salary_satisfaction > 0
                                      ? 'Well Paid'
                                      : 'Underpaid'}
                                  </span>
                                </div>
                                <div className="context-mini">
                                  <span className="context-icon">🦠</span>
                                  <span className="context-text">
                                    COVID: {existingEmployeeFactors.employee_data.covid_impact_score}/10
                                  </span>
                                </div>
                                <div className="context-mini">
                                  <span className="context-icon">📉</span>
                                  <span className="context-text">
                                    Economic: {existingEmployeeFactors.employee_data.economic_crisis_impact}/10
                                  </span>
                                </div>
                                <div className="context-mini">
                                  <span className="context-icon">🏛️</span>
                                  <span className="context-text">
                                    Political: {existingEmployeeFactors.employee_data.political_stability_concern}/10
                                  </span>
                                </div>
                              </div>
                            </div>
                          </>
                        ) : (
                          <div className="factors-placeholder">
                            <p>Loading factor analysis...</p>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
            </div>
          )}

          {employeeType === 'new' && (
            <div className="new-employee-section">
              <form onSubmit={handleSinglePrediction}>
                <div className="form-grid">
                  <div className="form-group">
                    <label>Employee Name (Optional)</label>
                    <input type="text" value={singleEmployee.employee_name} onChange={(e) => setSingleEmployee({...singleEmployee, employee_name: e.target.value})} placeholder="John Doe" />
                  </div>
                  <div className="form-group">
                    <label>Age *</label>
                    <input type="number" value={singleEmployee.age} onChange={(e) => setSingleEmployee({...singleEmployee, age: parseInt(e.target.value)})} min="22" max="65" required />
                  </div>
                  <div className="form-group">
                    <label>Job Role *</label>
                    <select value={singleEmployee.role} onChange={(e) => setSingleEmployee({...singleEmployee, role: e.target.value})} required>
                      {roles.map(role => (<option key={role} value={role}>{role}</option>))}
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Marital Status *</label>
                    <select value={singleEmployee.marital_status} onChange={(e) => setSingleEmployee({...singleEmployee, marital_status: e.target.value})} required>
                      {maritalStatuses.map(status => (<option key={status} value={status}>{status}</option>))}
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Work Experience (years) *</label>
                    <input type="number" step="0.1" value={singleEmployee.work_experience} onChange={(e) => setSingleEmployee({...singleEmployee, work_experience: parseFloat(e.target.value)})} min="0" max="40" required />
                  </div>
                  <div className="form-group">
                    <label>Time at Current Role (years) *</label>
                    <input type="number" step="0.1" value={singleEmployee.time_at_current_role} onChange={(e) => setSingleEmployee({...singleEmployee, time_at_current_role: parseFloat(e.target.value)})} min="0" max="20" required />
                  </div>
<div className="form-group">
                    <label>WFH Available *</label>
                    <select
                      value={singleEmployee.wfh_available}
                      onChange={(e) => setSingleEmployee({...singleEmployee, wfh_available: parseInt(e.target.value)})}
                      required
                    >
                      <option value={1}>Yes</option>
                      <option value={0}>No</option>
                    </select>
                  </div>
                </div>

                {/* Advanced Factors Toggle */}
                <div className="advanced-factors-toggle">
                  <button
                    type="button"
                    className="btn-toggle"
                    onClick={() => setShowAdvancedFactors(!showAdvancedFactors)}
                  >
                    {showAdvancedFactors ? '▼' : '▶'} Advanced Contextual Factors (Optional)
                  </button>
                  <p className="toggle-hint">
                    Leave blank to auto-calculate based on role and experience
                  </p>
                </div>

                {/* Advanced Factors Form */}
                {showAdvancedFactors && (
                  <div className="advanced-factors-form">
                    <h4>Contextual Factors</h4>
                    <p className="form-help">
                      These factors help predict attrition based on external circumstances.
                      If left blank, we'll calculate smart defaults.
                    </p>

                    <div className="form-grid">
                      <div className="form-group">
                        <label>
                          💰 Salary Satisfaction
                          <span className="label-hint">(-1 = Very Underpaid, 0 = Fair, 1 = Overpaid)</span>
                        </label>
                        <input
                          type="number"
                          step="0.1"
                          value={singleEmployee.salary_satisfaction || ''}
                          onChange={(e) => setSingleEmployee({
                            ...singleEmployee,
                            salary_satisfaction: e.target.value ? parseFloat(e.target.value) : null
                          })}
                          min="-1"
                          max="1"
                          placeholder="Auto-calculated if blank"
                        />
                      </div>

                      <div className="form-group">
                        <label>
                          🦠 COVID Impact Score
                          <span className="label-hint">(0-10, Higher = More Affected)</span>
                        </label>
                        <input
                          type="number"
                          step="0.1"
                          value={singleEmployee.covid_impact_score || ''}
                          onChange={(e) => setSingleEmployee({
                            ...singleEmployee,
                            covid_impact_score: e.target.value ? parseFloat(e.target.value) : null
                          })}
                          min="0"
                          max="10"
                          placeholder="Auto-calculated if blank"
                        />
                      </div>

                      <div className="form-group">
                        <label>
                          📉 Economic Crisis Impact
                          <span className="label-hint">(0-10, Sri Lanka 2022 Crisis)</span>
                        </label>
                        <input
                          type="number"
                          step="0.1"
                          value={singleEmployee.economic_crisis_impact || ''}
                          onChange={(e) => setSingleEmployee({
                            ...singleEmployee,
                            economic_crisis_impact: e.target.value ? parseFloat(e.target.value) : null
                          })}
                          min="0"
                          max="10"
                          placeholder="Auto-calculated if blank"
                        />
                      </div>

                      <div className="form-group">
                        <label>
                          🏛️ Political Instability Concern
                          <span className="label-hint">(0-10, Higher = More Concerned)</span>
                        </label>
                        <input
                          type="number"
                          step="0.1"
                          value={singleEmployee.political_stability_concern || ''}
                          onChange={(e) => setSingleEmployee({
                            ...singleEmployee,
                            political_stability_concern: e.target.value ? parseFloat(e.target.value) : null
                          })}
                          min="0"
                          max="10"
                          placeholder="Auto-calculated if blank"
                        />
                      </div>
                    </div>
                  </div>
                )}
                <button type="submit" className="btn-primary" disabled={loading}>{loading ? 'Predicting...' : 'Predict Attrition'}</button>
              </form>
{singlePrediction && (
  <div className="prediction-result-enhanced">
    <h3>Prediction Result</h3>

    <div className="result-layout">
      {/* Left side - Big percentage */}
      <div className="result-main">
        <div className="result-primary-card">
          <h4>Attrition Probability</h4>
          <div
            className="probability-big"
            style={{color: singlePrediction.risk_color}}
          >
            {singlePrediction.attrition_probability}%
          </div>
          <div className="risk-badge-large" style={{
            backgroundColor: singlePrediction.risk_color
          }}>
            {singlePrediction.risk_icon} {singlePrediction.risk_level}
          </div>
          <div className="prediction-meta">
            <div className="meta-item">
              <span className="meta-label">Will Leave?</span>
              <span className="meta-value">
                {singlePrediction.will_leave ? '⚠️ Likely' : '✅ Unlikely'}
              </span>
            </div>
            <div className="meta-item">
              <span className="meta-label">Confidence</span>
              <span className="meta-value">{singlePrediction.confidence}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Right side - Factors */}
      <div className="result-factors">
        {singleEmployeeFactors ? (
          <>
            <div className="factors-chart-small">
              <h4>Contributing Factors</h4>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart
                  data={singleEmployeeFactors.factors}
                  layout="vertical"
                  margin={{ left: 100, right: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 100]} />
                  <YAxis type="category" dataKey="name" width={90} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#3b82f6" radius={[0, 8, 8, 0]}>
                    {singleEmployeeFactors.factors.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="factors-list-small">
              {singleEmployeeFactors.factors.map((factor, idx) => (
                <div key={idx} className="factor-item-small">
                  <div className="factor-header-small">
                    <span
                      className="factor-dot"
                      style={{backgroundColor: factor.color}}
                    ></span>
                    <span className="factor-name">{factor.name}</span>
                    <span className="factor-value">{factor.value.toFixed(1)}%</span>
                  </div>
                </div>
              ))}
            </div>

            <div className="context-summary">
              <h4>Contextual Factors</h4>
              <div className="context-mini-grid">
                <div className="context-mini">
                  <span className="context-icon">💰</span>
                  <span className="context-text">
                    {singleEmployeeFactors.employee_data.salary_satisfaction > 0
                      ? 'Well Paid'
                      : 'Underpaid'}
                  </span>
                </div>
                <div className="context-mini">
                  <span className="context-icon">🦠</span>
                  <span className="context-text">
                    COVID: {singleEmployeeFactors.employee_data.covid_impact_score}/10
                  </span>
                </div>
                <div className="context-mini">
                  <span className="context-icon">📉</span>
                  <span className="context-text">
                    Economic: {singleEmployeeFactors.employee_data.economic_crisis_impact}/10
                  </span>
                </div>
                <div className="context-mini">
                  <span className="context-icon">🏛️</span>
                  <span className="context-text">
                    Political: {singleEmployeeFactors.employee_data.political_stability_concern}/10
                  </span>
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="factors-placeholder">
            <p>Loading factor analysis...</p>
          </div>
        )}
      </div>
    </div>
  </div>
)}
            </div>
          )}
        </div>
      )}

      {activeTab === 'department' && (
        <div className="department-prediction">
          <h2>Department Analysis</h2>
          <div className="department-selector">
            {['Engineering', 'QA', 'Business'].map(dept => (
              <button key={dept} className={`dept-btn ${selectedDepartment === dept ? 'active' : ''}`} onClick={() => handleDepartmentPrediction(dept)}>{dept}</button>
            ))}
          </div>
          {departmentPrediction && (
            <div className="department-results">
              <h3>{departmentPrediction.department} Department</h3>
              <p>Total Employees: {departmentPrediction.total_employees}</p>
              <div className="year-selector">
                {departmentPrediction.years.map(yearData => (
                  <button key={yearData.year} className={`year-btn ${deptSelectedYear === yearData.year ? 'active' : ''}`} onClick={() => setDeptSelectedYear(yearData.year)}>
                    <div className="year-label">{yearData.year}</div>
                    <div className="year-metric">{yearData.avg_attrition_probability}%</div>
                  </button>
                ))}
              </div>
              {deptSelectedYear && (
                <div className="quarter-breakdown">
                  <h4>Quarterly Breakdown for {deptSelectedYear}</h4>
                  <div className="quarters-grid">
                    {departmentPrediction.years.find(y => y.year === deptSelectedYear)?.quarters.map(quarter => (
                      <div key={quarter.quarter} className="quarter-card">
                        <h5>{quarter.quarter_label}</h5>
                        <div className="quarter-metric">{quarter.avg_attrition_probability}%</div>
                        <p className="quarter-leavers">Expected Leavers: {quarter.expected_leavers}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );

  const renderModal = () => {
    if (!showModal) return null;
    return (
      <div className="modal-overlay" onClick={() => setShowModal(false)}>
        <div className="modal-content" onClick={(e) => e.stopPropagation()}>
          <div className="modal-header">
            <h3>{modalData.title}</h3>
            <button className="modal-close" onClick={() => setShowModal(false)}>×</button>
          </div>
          <div className="modal-body">
            {modalData.employees.length > 0 ? (
              <div className="employee-list-modal">
                {modalData.employees.map((emp, idx) => (
                  <div key={idx} className="employee-item-modal">
                    <div className="employee-info-modal">
                      <strong>{emp.employee_id}</strong>
                      {emp.role !== 'N/A' && <span className="role-badge">{emp.role}</span>}
                      {emp.age !== 'N/A' && <span className="detail-text">Age: {emp.age}, Exp: {emp.work_experience}yr</span>}
                    </div>
                    <div className="risk-badge-modal" style={{backgroundColor: emp.attrition_probability >= 50 ? COLORS.veryHighRisk : emp.attrition_probability >= 30 ? COLORS.highRisk : COLORS.mediumRisk}}>
                      {emp.attrition_probability.toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="no-data">No employees found</p>
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderFactorModal = () => {
    if (!factorModal || !factorData) return null;
    return (
      <div className="modal-overlay" onClick={() => setFactorModal(false)}>
        <div className="modal-content modal-large" onClick={(e) => e.stopPropagation()}>
          <div className="modal-header">
            <div>
              <h3>Attrition Factor Analysis</h3>
              <p className="modal-subtitle">Employee: {factorData.employee_id} • Risk: {factorData.attrition_probability}%</p>
            </div>
            <button className="modal-close" onClick={() => setFactorModal(false)}>×</button>
          </div>
          <div className="modal-body">
            <div className="factor-analysis">
              <div className="factor-chart">
                <h4>Contributing Factors</h4>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={factorData.factors} layout="vertical" margin={{ left: 120 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[0, 100]} />
                    <YAxis type="category" dataKey="name" />
                    <Tooltip />
                    <Bar dataKey="value" fill="#3b82f6" radius={[0, 8, 8, 0]}>
                      {factorData.factors.map((entry, index) => (<Cell key={`cell-${index}`} fill={entry.color} />))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="factor-details">
                <h4>Factor Breakdown</h4>
                {factorData.factors.map((factor, idx) => (
                  <div key={idx} className="factor-item">
                    <div className="factor-header">
                      <span className="factor-dot" style={{backgroundColor: factor.color}}></span>
                      <strong>{factor.name}</strong>
                      <span className="factor-percentage">{factor.value.toFixed(1)}%</span>
                    </div>
                    <p className="factor-description">{factor.description}</p>
                  </div>
                ))}
              </div>
              <div className="employee-context">
                <h4>Employee Context</h4>
                <div className="context-grid">
                  <div className="context-item">
                    <label>Salary Satisfaction</label>
                    <div className="context-value">{factorData.employee_data.salary_satisfaction > 0 ? '✅ Well Paid' : factorData.employee_data.salary_satisfaction < -0.3 ? '🔴 Significantly Underpaid' : '⚠️ Slightly Underpaid'}</div>
                  </div>
                  <div className="context-item">
                    <label>COVID Impact</label>
                    <div className="context-value">{factorData.employee_data.covid_impact_score}/10</div>
                  </div>
                  <div className="context-item">
                    <label>Economic Crisis Impact</label>
                    <div className="context-value">{factorData.employee_data.economic_crisis_impact}/10</div>
                  </div>
                  <div className="context-item">
                    <label>Political Concern</label>
                    <div className="context-value">{factorData.employee_data.political_stability_concern}/10</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

const renderInsights = () => (
  <div className="insights-container">
    <div className="insights-header">
      <h1>AI-Powered Insights</h1>
      <p className="insights-subtitle">Advanced analytics and retention strategies</p>
    </div>

    {!uploadedAnalytics ? (
      <div className="empty-state">
        <h2>No Data Available</h2>
        <p>Please upload employee data from the Home page to view insights</p>
        <button onClick={() => setActiveTab('home')} className="btn-primary">
          Go to Home
        </button>
      </div>
    ) : (
      <>
        <div className="insights-tabs">
          <button
            className={insightsTab === 'optimizer' ? 'tab-btn active' : 'tab-btn'}
            onClick={() => setInsightsTab('optimizer')}
          >
            🎯 Retention Optimizer
          </button>
          <button
            className={insightsTab === 'clustering' ? 'tab-btn active' : 'tab-btn'}
            onClick={() => {
              setInsightsTab('clustering');
              if (!clusteringData) fetchClusteringData();
            }}
          >
            🔬 Risk Profiles
          </button>
        </div>

        {insightsTab === 'optimizer' && (
          <div className="optimizer-section">
            {!selectedEmployeeForStrategy ? (
              <div className="employee-selector-section">
                <h2>Select High-Risk Employee</h2>
                <p>Choose an employee to generate personalized retention strategies</p>

                <div className="employee-grid">
                  {uploadedAnalytics.high_risk_employees?.slice(0, 20).map((emp, idx) => (
                    <div
                      key={idx}
                      className="employee-select-card"
                      onClick={() => handleSelectEmployeeForStrategy(emp.employee_id)}
                    >
                      <div className="emp-card-header">
                        <strong>{emp.employee_id}</strong>
                        <span className="emp-risk" style={{
                          backgroundColor: emp.attrition_probability >= 60 ? COLORS.veryHighRisk : COLORS.highRisk
                        }}>
                          {emp.attrition_probability.toFixed(0)}%
                        </span>
                      </div>
                      <div className="emp-card-body">
                        <p>{emp.role}</p>
                        <small>Age: {emp.age} • {emp.work_experience}yr exp</small>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="strategies-section">
                <button
                  className="btn-back"
                  onClick={() => {
                    setSelectedEmployeeForStrategy(null);
                    setRetentionStrategies(null);
                  }}
                >
                  ← Back to Employee Selection
                </button>

                {loadingInsights ? (
                  <div className="loading-state">
                    <p>⏳ Generating AI-powered retention strategies...</p>
                  </div>
                ) : retentionStrategies ? (
                  <>
                    <div className="strategies-header">
                      <h2>Retention Strategies for {selectedEmployeeForStrategy}</h2>
                      <div className="current-risk-badge">
                        Current Risk: <strong>{retentionStrategies.current_risk}%</strong>
                      </div>
                    </div>

                    <div className="strategies-grid">
                      {retentionStrategies.strategies.map((strategy, idx) => (
                        <div key={idx} className="strategy-card">
                          <div className="strategy-header">
                            <span className="strategy-icon">{strategy.icon}</span>
                            <h3>{strategy.name}</h3>
                            {strategy.priority === 1 && (
                              <span className="priority-badge">High Priority</span>
                            )}
                          </div>

                          <div className="strategy-metrics">
                            <div className="metric-row">
                              <span className="metric-label">Current Risk</span>
                              <span className="metric-value risk-current">
                                {strategy.current_risk}%
                              </span>
                            </div>
                            <div className="metric-arrow">→</div>
                            <div className="metric-row">
                              <span className="metric-label">New Risk</span>
                              <span className="metric-value risk-new">
                                {strategy.new_risk}%
                              </span>
                            </div>
                            <div className="metric-reduction">
                              <span className="reduction-badge">
                                ⬇️ {strategy.reduction}%
                              </span>
                            </div>
                          </div>

                          <p className="strategy-description">{strategy.description}</p>

                          <div className="strategy-financials">
                            <div className="financial-item">
                              <span className="fin-label">Investment</span>
                              <span className="fin-value cost">
                                ${(strategy.cost / 1000).toFixed(0)}K
                              </span>
                            </div>
                            <div className="financial-item">
                              <span className="fin-label">Potential Savings</span>
                              <span className="fin-value savings">
                                ${(strategy.savings / 1000).toFixed(0)}K
                              </span>
                            </div>
                            <div className="financial-item highlight">
                              <span className="fin-label">ROI</span>
                              <span className="fin-value roi">
                                {strategy.roi}%
                              </span>
                            </div>
                          </div>

                          <button className="btn-apply-strategy">
                            Apply Strategy
                          </button>
                        </div>
                      ))}
                    </div>

                    <div className="combined-impact-card">
                      <h3>📊 Combined Impact Analysis</h3>
                      <p>Implementing top 3 strategies together:</p>

                      <div className="combined-metrics">
                        <div className="combined-metric">
                          <span className="cm-label">Projected Risk</span>
                          <span className="cm-value">
                            {retentionStrategies.current_risk}% → {retentionStrategies.combined_impact.implementing_top_3}%
                          </span>
                        </div>
                        <div className="combined-metric">
                          <span className="cm-label">Total Investment</span>
                          <span className="cm-value">
                            ${(retentionStrategies.combined_impact.total_cost / 1000).toFixed(0)}K
                          </span>
                        </div>
                        <div className="combined-metric">
                          <span className="cm-label">Expected Savings</span>
                          <span className="cm-value">
                            ${(retentionStrategies.combined_impact.total_savings / 1000).toFixed(0)}K
                          </span>
                        </div>
                        <div className="combined-metric highlight">
                          <span className="cm-label">Net Benefit</span>
                          <span className="cm-value">
                            ${(retentionStrategies.combined_impact.net_benefit / 1000).toFixed(0)}K
                          </span>
                        </div>
                      </div>

                      <button className="btn-generate-report">
                        📄 Generate Executive Report
                      </button>
                    </div>
                  </>
                ) : null}
              </div>
            )}
          </div>
        )}

        {insightsTab === 'clustering' && (
          <div className="clustering-section">
            {loadingInsights ? (
              <div className="loading-state">
                <p>⏳ Performing ML clustering analysis...</p>
              </div>
            ) : clusteringData ? (
              <>
                <div className="clustering-header">
                  <h2>Employee Risk Profiles</h2>
                  <p>AI-identified patterns using K-Means clustering</p>

                  <div className="clustering-stats">
                    <div className="stat-box">
                      <span className="stat-value">{clusteringData.total_employees}</span>
                      <span className="stat-label">Total Employees</span>
                    </div>
                    <div className="stat-box">
                      <span className="stat-value">{clusteringData.clusters.length}</span>
                      <span className="stat-label">Risk Profiles</span>
                    </div>
                    <div className="stat-box warning">
                      <span className="stat-value">{clusteringData.early_warnings.anomalies_detected}</span>
                      <span className="stat-label">⚠️ Anomalies Detected</span>
                    </div>
                  </div>
                </div>

                {clusteringData.early_warnings.anomalies_detected > 0 && (
                  <div className="alert-card">
                    <h3>🚨 Early Warning System Alert</h3>
                    <p>Detected {clusteringData.early_warnings.anomalies_detected} employees with unusual risk patterns</p>
                    <div className="anomaly-list">
                      {clusteringData.early_warnings.sudden_risk_employees.map((emp, idx) => (
                        <div key={idx} className="anomaly-item">
                          <strong>{emp.employee_id}</strong>
                          <span>{emp.role}</span>
                          <span className="anomaly-risk">{emp.attrition_probability.toFixed(0)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="clusters-grid">
                  {clusteringData.clusters.map((cluster, idx) => (
                    <div
                      key={idx}
                      className={`cluster-card ${cluster.risk_level.toLowerCase()}`}
                      onClick={() => setSelectedCluster(selectedCluster === cluster.id ? null : cluster.id)}
                    >
                      <div className="cluster-header">
                        <h3>{cluster.name}</h3>
                        <span className={`risk-level-badge ${cluster.risk_level.toLowerCase()}`}>
                          {cluster.risk_level}
                        </span>
                      </div>

                      <div className="cluster-stats">
                        <div className="cluster-stat">
                          <span className="cs-label">Employees</span>
                          <span className="cs-value">{cluster.size}</span>
                        </div>
                        <div className="cluster-stat">
                          <span className="cs-label">Avg Risk</span>
                          <span className="cs-value">{cluster.avg_risk}%</span>
                        </div>
                      </div>

                      <div className="cluster-factors">
                        <strong>Key Factors:</strong>
                        <ul>
                          {cluster.key_factors.map((factor, fidx) => (
                            <li key={fidx}>{factor}</li>
                          ))}
                        </ul>
                      </div>

                      {selectedCluster === cluster.id && (
                        <div className="cluster-details">
                          <div className="cluster-characteristics">
                            <h4>Profile Characteristics:</h4>
                            <div className="char-grid">
                              <div className="char-item">
                                <span>Avg Age:</span>
                                <strong>{cluster.characteristics.avg_age}</strong>
                              </div>
                              <div className="char-item">
                                <span>Avg Experience:</span>
                                <strong>{cluster.characteristics.avg_experience}yr</strong>
                              </div>
                              <div className="char-item">
                                <span>COVID Impact:</span>
                                <strong>{cluster.characteristics.avg_covid_impact}/10</strong>
                              </div>
                              <div className="char-item">
                                <span>Economic Impact:</span>
                                <strong>{cluster.characteristics.avg_economic_impact}/10</strong>
                              </div>
                            </div>
                          </div>

                          <div className="cluster-actions">
                            <h4>Recommended Actions:</h4>
                            <ul className="action-list">
                              {cluster.recommended_actions.map((action, aidx) => (
                                <li key={aidx}>✓ {action}</li>
                              ))}
                            </ul>
                          </div>

                          <div className="cluster-top-employees">
                            <h4>Top 5 Risk Employees:</h4>
                            <div className="top-emp-list">
                              {cluster.top_employees.map((emp, eidx) => (
                                <div key={eidx} className="top-emp-item">
                                  <span className="emp-id">{emp.employee_id}</span>
                                  <span className="emp-role">{emp.role}</span>
                                  <span className="emp-risk">{emp.attrition_probability.toFixed(0)}%</span>
                                </div>
                              ))}
                            </div>
                          </div>

                          <button
                            className="btn-cluster-strategy"
                            onClick={() => handleSelectEmployeeForStrategy(cluster.top_employees[0].employee_id)}
                          >
                            Generate Strategies for This Profile
                          </button>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="empty-state">
                <button onClick={fetchClusteringData} className="btn-primary">
                  🔬 Run Clustering Analysis
                </button>
              </div>
            )}
          </div>
        )}
      </>
    )}
  </div>
);
  const renderReports = () => (<div className="reports-container"><h2>Reports</h2><p>Coming soon - Export and detailed reports</p></div>);

  return (
    <div className="App">
      <div className="sidebar">
        <div className="logo"><h2>WSM</h2><p>Workforce Strategy Model</p></div>
        <nav>
          <button className={activeTab === 'home' ? 'nav-item active' : 'nav-item'} onClick={() => setActiveTab('home')}>🏠 Home</button>
          <button className={activeTab === 'single' || activeTab === 'department' ? 'nav-item active' : 'nav-item'} onClick={() => setActiveTab('single')}>🎯 Prediction</button>
          <button className={activeTab === 'insights' ? 'nav-item active' : 'nav-item'} onClick={() => setActiveTab('insights')}>💡 Insights</button>
          <button className={activeTab === 'reports' ? 'nav-item active' : 'nav-item'} onClick={() => setActiveTab('reports')}>📊 Reports</button>
        </nav>
      </div>
      <div className="main-content">
        <header className="top-header">
          <h1>Employee Turnover Prediction</h1>
          <div className="header-actions">
            <button className="btn-header">⚙️ Settings</button>
            <button className="btn-header">👤 Profile</button>
          </div>
        </header>
        <div className="content-area">
          {activeTab === 'home' && renderHome()}
          {(activeTab === 'single' || activeTab === 'department') && renderPrediction()}
          {activeTab === 'insights' && renderInsights()}
          {activeTab === 'reports' && renderReports()}
        </div>
      </div>
      {renderModal()}
      {renderFactorModal()}
    </div>
  );
}

export default App;