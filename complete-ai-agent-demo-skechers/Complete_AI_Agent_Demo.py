# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # End-to-End AI Agent with Claude Sonnet 4.5, RAG, and Microsoft Teams Integration
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/ai-agent/agent-demo-0.png?raw=true" width="800px" style="float: right">
# MAGIC
# MAGIC ## Overview
# MAGIC This comprehensive demo showcases the complete Databricks AI Agent Framework capabilities:
# MAGIC
# MAGIC 1. **LLM Setup** - Claude Sonnet 4.5 as the foundation model
# MAGIC 2. **Multiple Knowledge Sources** - 3 different Vector Search indexes for RAG
# MAGIC 3. **Prompt Engineering** - Iterative prompt optimization
# MAGIC 4. **Agent Creation** - LangGraph-based agent with tools
# MAGIC 5. **Agent Evaluation** - MLflow evaluation with custom scorers
# MAGIC 6. **Model Serving** - Deploy agent to Model Serving endpoint
# MAGIC 7. **Dynamic System Prompts** - Runtime prompt customization
# MAGIC 8. **Microsoft Teams Integration** - Serve responses to Teams
# MAGIC 9. **Feedback Loop** - Monitor emoji reactions and log to MLflow
# MAGIC
# MAGIC ### Key Databricks Features Demonstrated:
# MAGIC - **Unity Catalog Functions** as tools
# MAGIC - **Vector Search** for knowledge retrieval
# MAGIC - **MLflow 3.0** for tracing and evaluation
# MAGIC - **MLflow Evaluation** with custom scorers and guidelines
# MAGIC - **Model Serving** with dynamic configuration
# MAGIC - **Agent Framework** for orchestration

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install -U -qqqq mlflow>=3.1.4 langchain>=1.0.0 langgraph databricks-langchain pydantic databricks-agents unitycatalog-langchain[databricks] uv databricks-feature-engineering==0.12.1 pymsteams requests
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configuration
# Set your configuration
catalog = "amitabh_arora_catalog"
schema = dbName = db = "demo_im_agent"
volume_name = "raw_data"

# Vector Search endpoint
VECTOR_SEARCH_ENDPOINT_NAME = "demo_im_agent_vs_endpoint"

# Model configuration
MODEL_NAME = "demo_im_agent_claude_sonnet"
ENDPOINT_NAME = f'{catalog}_{MODEL_NAME}'[:60]

# LLM Model - Claude Sonnet 4.5
LLM_ENDPOINT_NAME = 'databricks-claude-sonnet-4-5'

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Vector Search Endpoint: {VECTOR_SEARCH_ENDPOINT_NAME}")
print(f"Model Serving Endpoint: {ENDPOINT_NAME}")
print(f"LLM Endpoint: {LLM_ENDPOINT_NAME}")

# COMMAND ----------

# DBTITLE 1,Setup Schema and Volume
import sys
major, minor = sys.version_info[:2]
assert (major, minor) >= (3, 9), f"This demo requires Python 3.9+, but found {major}.{minor}"

# Setup catalog and schema
spark.sql(f"CREATE CATALOG IF NOT EXISTS `{catalog}`")
spark.sql(f"USE CATALOG `{catalog}`")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{schema}`")
spark.sql(f"USE SCHEMA `{schema}`")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {volume_name}")

volume_folder = f"/Volumes/{catalog}/{schema}/{volume_name}"
print(f"Volume location: {volume_folder}")

# COMMAND ----------

# DBTITLE 1,Helper Functions for Vector Search
import time

def endpoint_exists(vsc, vs_endpoint_name):
    try:
        return vs_endpoint_name in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]
    except Exception as e:
        if "REQUEST_LIMIT_EXCEEDED" in str(e):
            print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error.")
            return True
        else:
            raise e

def wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name):
    for i in range(180):
        try:
            endpoint = vsc.get_endpoint(vs_endpoint_name)
        except Exception as e:
            if "REQUEST_LIMIT_EXCEEDED" in str(e):
                print("WARN: couldn't get endpoint status. Please manually check.")
                return
            else:
                raise e
        status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
        if "ONLINE" in status:
            return endpoint
        elif "PROVISIONING" in status or i < 6:
            if i % 2 == 0:
                print(f"Waiting for endpoint to be ready... {status}")
            time.sleep(10)
        else:
            raise Exception(f'Error with endpoint {vs_endpoint_name}: {endpoint}')
    raise Exception(f"Timeout waiting for endpoint: {vs_endpoint_name}")

def index_exists(vsc, endpoint_name, index_full_name):
    try:
        vsc.get_index(endpoint_name, index_full_name).describe()
        return True
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            print(f'Unexpected error describing index.')
            raise e
    return False

def wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name):
    for i in range(180):
        idx = vsc.get_index(vs_endpoint_name, index_name).describe()
        index_status = idx.get('status', idx.get('index_status', {}))
        status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
        url = index_status.get('index_url', index_status.get('url', 'UNKNOWN'))
        if "ONLINE" in status:
            return
        if "UNKNOWN" in status:
            print(f"Can't get status - assuming ready: {url}")
            return
        elif "PROVISIONING" in status:
            if i % 40 == 0:
                print(f"Waiting for index... {status} - {url}")
            time.sleep(10)
        else:
            raise Exception(f'Error with index {index_name}: {idx}')
    raise Exception(f"Timeout waiting for index: {index_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Prepare Sample Data
# MAGIC
# MAGIC We'll create sample data tables that our agent will query using Unity Catalog functions.

# COMMAND ----------

# DBTITLE 1,Create Sample Customer Data
# Create customers table
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog}.{schema}.customers (
    customer_id STRING,
    email STRING,
    name STRING,
    phone STRING,
    address STRING,
    city STRING,
    state STRING,
    zip_code STRING,
    account_status STRING
) USING DELTA
""")

# Insert sample data
spark.sql(f"""
INSERT INTO {catalog}.{schema}.customers VALUES
    ('C001', 'john21@example.net', 'John Smith', '555-0101', '123 Main St', 'San Francisco', 'CA', '94102', 'active'),
    ('C002', 'sarah.jones@example.com', 'Sarah Jones', '555-0102', '456 Oak Ave', 'San Jose', 'CA', '95110', 'active'),
    ('C003', 'mike.brown@example.com', 'Mike Brown', '555-0103', '789 Pine Rd', 'Oakland', 'CA', '94601', 'active'),
    ('C004', 'emily.davis@example.com', 'Emily Davis', '555-0104', '321 Elm St', 'Berkeley', 'CA', '94704', 'suspended')
""")

print("✅ Customers table created")

# COMMAND ----------

# DBTITLE 1,Create Sample Billing Data
# Create billing table
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog}.{schema}.billing (
    bill_id STRING,
    customer_id STRING,
    bill_date DATE,
    amount DOUBLE,
    status STRING,
    due_date DATE,
    service_type STRING
) USING DELTA
""")

# Insert sample data
spark.sql(f"""
INSERT INTO {catalog}.{schema}.billing VALUES
    ('B001', 'C001', '2024-01-01', 89.99, 'paid', '2024-01-15', 'Internet'),
    ('B002', 'C001', '2024-02-01', 89.99, 'paid', '2024-02-15', 'Internet'),
    ('B003', 'C002', '2024-01-01', 129.99, 'paid', '2024-01-15', 'Internet + TV'),
    ('B004', 'C003', '2024-01-01', 59.99, 'overdue', '2024-01-15', 'Internet'),
    ('B005', 'C004', '2024-01-01', 149.99, 'pending', '2024-01-15', 'Premium Package')
""")

print("✅ Billing table created")

# COMMAND ----------

# DBTITLE 1,Create Sample Support Tickets
# Create support tickets table
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog}.{schema}.support_tickets (
    ticket_id STRING,
    customer_id STRING,
    issue_type STRING,
    description STRING,
    status STRING,
    created_date DATE,
    resolved_date DATE
) USING DELTA
""")

# Insert sample data
spark.sql(f"""
INSERT INTO {catalog}.{schema}.support_tickets VALUES
    ('T001', 'C001', 'connection', 'Slow internet speed', 'resolved', '2024-01-05', '2024-01-06'),
    ('T002', 'C002', 'billing', 'Incorrect charge on bill', 'open', '2024-02-10', NULL),
    ('T003', 'C003', 'technical', 'Router not connecting', 'in_progress', '2024-02-15', NULL),
    ('T004', 'C004', 'account', 'Account suspension inquiry', 'open', '2024-02-20', NULL)
""")

print("✅ Support tickets table created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Unity Catalog Functions as Agent Tools
# MAGIC
# MAGIC These functions will be used as tools by our agent.
# MAGIC

# COMMAND ----------

spark.sql(f"DROP FUNCTION IF EXISTS {catalog}.{schema}.get_customer_info")
spark.sql(f"DROP FUNCTION IF EXISTS {catalog}.{schema}.get_support_tickets")
spark.sql(f"DROP FUNCTION IF EXISTS {catalog}.{schema}.get_billing_info")

# COMMAND ----------

# DBTITLE 1,Function 1: Get Customer Information

spark.sql(f"""
CREATE OR REPLACE FUNCTION {catalog}.{schema}.get_customer_info(
  customer_email STRING COMMENT 'The email address of the customer to look up. Must be a valid email format (e.g., john@example.com).'
)
RETURNS STRING
COMMENT 'Retrieves customer information by email address including customer ID, name, phone number, account status, and full address'
RETURN 
  CASE 
    WHEN EXISTS (SELECT 1 FROM {catalog}.{schema}.customers WHERE email = customer_email) THEN
      (SELECT CONCAT(
        'Customer ID: ', customer_id, ', ',
        'Name: ', name, ', ',
        'Phone: ', phone, ', ',
        'Status: ', account_status, ', ',
        'Address: ', address, ', ', city, ', ', state, ' ', zip_code
      ) FROM {catalog}.{schema}.customers WHERE email = customer_email LIMIT 1)
    ELSE
      CONCAT('No customer found with email: ', customer_email)
  END
""")

print("✅ Function created: get_customer_info")

# COMMAND ----------

# DBTITLE 1,Function 2: Get Billing Information
spark.sql(f"""
CREATE OR REPLACE FUNCTION {catalog}.{schema}.get_billing_info(
  customer_email STRING COMMENT 'The email address of the customer whose billing information to retrieve. Must be a valid email format (e.g., john@example.com).'
)
RETURNS STRING
COMMENT 'Retrieves all billing records for a customer by email address, including bill ID, amount, payment status, due date, and service type. Returns bills sorted by most recent first.'
RETURN 
  CASE 
    WHEN NOT EXISTS (SELECT 1 FROM {catalog}.{schema}.customers WHERE email = customer_email) THEN
      CONCAT('No customer found with email: ', customer_email)
    WHEN NOT EXISTS (
      SELECT 1 FROM {catalog}.{schema}.billing b
      JOIN {catalog}.{schema}.customers c ON b.customer_id = c.customer_id
      WHERE c.email = customer_email
    ) THEN
      CONCAT('No billing records found for customer: ', customer_email)
    ELSE
      (SELECT CONCAT_WS('\\n',
        COLLECT_LIST(bill_info)
      )
      FROM (
        SELECT CONCAT('Bill ', bill_id, ': $', CAST(amount AS STRING), ' - ', status, 
                      ' (Due: ', CAST(due_date AS STRING), ') - ', service_type) as bill_info
        FROM {catalog}.{schema}.billing b
        JOIN {catalog}.{schema}.customers c ON b.customer_id = c.customer_id
        WHERE c.email = customer_email
        ORDER BY bill_date DESC
      ))
  END
""")

print("✅ Function created: get_billing_info")

# COMMAND ----------

# DBTITLE 1,Function 3: Get Support Tickets
spark.sql(f"""
CREATE OR REPLACE FUNCTION {catalog}.{schema}.get_support_tickets(
  customer_email STRING COMMENT 'The email address of the customer whose support tickets to retrieve. Must be a valid email format (e.g., john@example.com).'
)
RETURNS STRING
COMMENT 'Retrieves all support tickets for a customer by email address, including ticket ID, issue type, description, and current status or resolution date. Returns tickets sorted by most recent first.'
RETURN 
  CASE 
    WHEN NOT EXISTS (SELECT 1 FROM {catalog}.{schema}.customers WHERE email = customer_email) THEN
      CONCAT('No customer found with email: ', customer_email)
    WHEN NOT EXISTS (
      SELECT 1 FROM {catalog}.{schema}.support_tickets t
      JOIN {catalog}.{schema}.customers c ON t.customer_id = c.customer_id
      WHERE c.email = customer_email
    ) THEN
      CONCAT('No support tickets found for customer: ', customer_email)
    ELSE
      (SELECT CONCAT_WS('\\n',
        COLLECT_LIST(ticket_info)
      )
      FROM (
        SELECT CONCAT('Ticket ', ticket_id, ': ', issue_type, ' - ', description, ' (',
                      CASE 
                        WHEN status = 'resolved' THEN CONCAT('Resolved: ', CAST(resolved_date AS STRING))
                        ELSE CONCAT('Status: ', status)
                      END, ')') as ticket_info
        FROM {catalog}.{schema}.support_tickets t
        JOIN {catalog}.{schema}.customers c ON t.customer_id = c.customer_id
        WHERE c.email = customer_email
        ORDER BY created_date DESC
      ))
  END
""")

print("✅ Function created: get_support_tickets")

# COMMAND ----------

# DBTITLE 1,Test Unity Catalog Functions
# Test the functions
print("Testing get_customer_info:")
result = spark.sql(f"SELECT {catalog}.{schema}.get_customer_info('john21@example.net') as result").first().result
print(result)

print("\nTesting get_billing_info:")
result = spark.sql(f"SELECT {catalog}.{schema}.get_billing_info('john21@example.net') as result").first().result
print(result)

print("\nTesting get_support_tickets:")
result = spark.sql(f"SELECT {catalog}.{schema}.get_support_tickets('john21@example.net') as result").first().result
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Multiple Vector Search Indexes for RAG
# MAGIC
# MAGIC We'll create 3 different knowledge bases:
# MAGIC 1. Product documentation (routers, equipment)
# MAGIC 2. Technical troubleshooting guides
# MAGIC 3. Policy and compliance documents

# COMMAND ----------

# DBTITLE 1,Setup Vector Search Endpoint
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

# Create Vector Search endpoint if it doesn't exist
if not endpoint_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME):
    print(f"Creating Vector Search endpoint: {VECTOR_SEARCH_ENDPOINT_NAME}")
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")
    wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
else:
    print(f"Vector Search endpoint already exists: {VECTOR_SEARCH_ENDPOINT_NAME}")

print(f"✅ Endpoint {VECTOR_SEARCH_ENDPOINT_NAME} is ready")

# COMMAND ----------

# DBTITLE 1,Knowledge Base 1: Product Documentation
# Create product documentation table
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog}.{schema}.product_docs (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    product_name STRING,
    title STRING,
    content STRING,
    doc_type STRING
) TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# Insert sample product documentation
spark.sql(f"""
INSERT INTO {catalog}.{schema}.product_docs (product_name, title, content, doc_type) VALUES
    ('ADSL-R500', 'ADSL-R500 Router Setup Guide', 'The ADSL-R500 is a high-performance router. To restart: 1) Unplug power cable 2) Wait 30 seconds 3) Plug back in 4) Wait for all lights to turn green. For Error Code 01, check DSL connection and ensure cables are firmly connected.', 'setup'),
    ('FIBER-X1000', 'FIBER-X1000 Installation Guide', 'The FIBER-X1000 is our premium fiber router. Installation: 1) Connect fiber cable to WAN port 2) Connect power adapter 3) Connect to device via WiFi or Ethernet. Default WiFi password is on the label underneath.', 'installation'),
    ('MESH-PRO-300', 'MESH-PRO-300 Mesh Network Setup', 'The MESH-PRO-300 extends your network coverage. Setup: 1) Plug in near your main router 2) Press WPS button on both devices 3) Wait for solid green light. Each unit covers up to 1500 sq ft.', 'setup')
""")

print("✅ Product documentation table created")

# Create Vector Search index for product docs
product_docs_index = f"{catalog}.{schema}.product_docs_index"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, product_docs_index):
    print(f"Creating Vector Search index: {product_docs_index}")
    vsc.create_delta_sync_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=product_docs_index,
        source_table_name=f"{catalog}.{schema}.product_docs",
        pipeline_type="TRIGGERED",
        primary_key="id",
        embedding_source_column='content',
        embedding_model_endpoint_name='databricks-gte-large-en'
    )
    wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, product_docs_index)
else:
    vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, product_docs_index).sync()

print(f"✅ Product docs index ready: {product_docs_index}")

# COMMAND ----------

# DBTITLE 1,Knowledge Base 2: Technical Troubleshooting
# Create troubleshooting table
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog}.{schema}.troubleshooting_docs (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    issue_type STRING,
    title STRING,
    content STRING,
    severity STRING
) TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# Insert sample troubleshooting documentation
spark.sql(f"""
INSERT INTO {catalog}.{schema}.troubleshooting_docs (issue_type, title, content, severity) VALUES
    ('connectivity', 'Slow Internet Speed Troubleshooting', 'For slow internet: 1) Run speed test at speedtest.net 2) Restart router 3) Check for interference from other devices 4) Try wired connection to rule out WiFi issues 5) Contact support if speeds are below 80% of plan speed.', 'medium'),
    ('error_codes', 'Router Error Code Reference', 'Error Code 01: DSL sync failure - check phone line connection. Error Code 02: Authentication failure - verify account credentials. Error Code 03: Hardware malfunction - contact support for replacement. Error Code 04: Firmware update required - system will auto-update.', 'high'),
    ('wifi_issues', 'WiFi Connection Problems', 'WiFi not connecting: 1) Verify password is correct 2) Forget network and reconnect 3) Update device drivers 4) Change WiFi channel in router settings 5) Reduce distance to router. For 5GHz issues, note that range is shorter than 2.4GHz.', 'medium')
""")

print("✅ Troubleshooting documentation table created")

# Create Vector Search index for troubleshooting docs
troubleshooting_index = f"{catalog}.{schema}.troubleshooting_docs_index"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, troubleshooting_index):
    print(f"Creating Vector Search index: {troubleshooting_index}")
    vsc.create_delta_sync_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=troubleshooting_index,
        source_table_name=f"{catalog}.{schema}.troubleshooting_docs",
        pipeline_type="TRIGGERED",
        primary_key="id",
        embedding_source_column='content',
        embedding_model_endpoint_name='databricks-gte-large-en'
    )
    wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, troubleshooting_index)
else:
    vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, troubleshooting_index).sync()

print(f"✅ Troubleshooting index ready: {troubleshooting_index}")

# COMMAND ----------

# DBTITLE 1,Knowledge Base 3: Policy and Compliance (with Confluence Metadata)
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog}.{schema}.policy_docs (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    policy_type STRING,
    title STRING,
    content STRING,
    content_html STRING,
    effective_date DATE,
    source_url STRING,
    confluence_page_id STRING,
    last_updated TIMESTAMP,
    tags ARRAY<STRING>
) TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

spark.sql(f"""
INSERT INTO {catalog}.{schema}.policy_docs 
(policy_type, title, content, content_html, effective_date, source_url, confluence_page_id, last_updated, tags) 
VALUES
    (
        'billing', 
        'Billing Dispute Policy', 
        'Customers can dispute charges within 30 days of billing date. To dispute: 1) Contact customer support 2) Provide bill number and disputed amount 3) Explain reason for dispute. Resolution typically takes 5-7 business days. Credits are applied to next bill cycle.',
        '<h2>Billing Dispute Policy</h2><p>Customers can dispute charges within <strong>30 days</strong> of billing date.</p><h3>Dispute Process:</h3><ol><li>Contact customer support</li><li>Provide bill number and disputed amount</li><li>Explain reason for dispute</li></ol><p>Resolution typically takes <strong>5-7 business days</strong>. Credits are applied to next bill cycle.</p>',
        '2024-01-01',
        'https://confluence.company.com/display/POLICY/Billing+Dispute+Policy',
        'POLICY-12345',
        CURRENT_TIMESTAMP(),
        array('billing', 'disputes', 'customer-service', 'finance')
    ),
    (
        'refund', 
        'Service Cancellation and Refund Policy', 
        'Customers may cancel service anytime with 30 days notice. Early termination fees apply if under contract. Refunds are prorated for unused service time. Equipment must be returned within 14 days to avoid equipment charges ($150 router, $100 modem).',
        '<h2>Service Cancellation and Refund Policy</h2><p>Customers may cancel service anytime with <strong>30 days notice</strong>.</p><h3>Important Notes:</h3><ul><li>Early termination fees apply if under contract</li><li>Refunds are prorated for unused service time</li><li>Equipment must be returned within <strong>14 days</strong> to avoid charges:</li></ul><table><tr><th>Equipment</th><th>Charge</th></tr><tr><td>Router</td><td>$150</td></tr><tr><td>Modem</td><td>$100</td></tr></table>',
        '2024-01-01',
        'https://confluence.company.com/display/POLICY/Service+Cancellation+and+Refund',
        'POLICY-12346',
        CURRENT_TIMESTAMP(),
        array('refund', 'cancellation', 'equipment', 'customer-service')
    ),
    (
        'privacy', 
        'Data Privacy and Usage Policy', 
        'We collect service usage data to improve quality. Personal data is never sold to third parties. Customers can request data deletion per GDPR/CCPA. Network traffic is monitored for security but content is not inspected. Logs are retained for 90 days.',
        '<h2>Data Privacy and Usage Policy</h2><p>We collect service usage data to improve quality. <strong>Personal data is never sold to third parties.</strong></p><h3>Customer Rights:</h3><ul><li>Request data deletion per GDPR/CCPA</li><li>Access personal data on file</li><li>Opt-out of non-essential data collection</li></ul><h3>Data Retention:</h3><p>Network traffic is monitored for security but <em>content is not inspected</em>. Logs are retained for <strong>90 days</strong>.</p>',
        '2024-01-01',
        'https://confluence.company.com/display/POLICY/Data+Privacy+and+Usage',
        'POLICY-12347',
        CURRENT_TIMESTAMP(),
        array('privacy', 'gdpr', 'ccpa', 'data-retention', 'security')
    ),
    (
        'sla',
        'Service Level Agreement (SLA)',
        'We guarantee 99.9% uptime for internet services. Scheduled maintenance windows occur monthly. Customers receive credits for outages exceeding 4 hours: 10% credit for 4-8 hours, 25% credit for 8-24 hours, 50% credit for 24+ hours. SLA applies to business accounts only.',
        '<h2>Service Level Agreement (SLA)</h2><p>We guarantee <strong>99.9% uptime</strong> for internet services.</p><h3>Maintenance Windows:</h3><p>Scheduled maintenance occurs <em>monthly</em> with 48-hour advance notice.</p><h3>Service Credits:</h3><table><tr><th>Outage Duration</th><th>Credit</th></tr><tr><td>4-8 hours</td><td>10%</td></tr><tr><td>8-24 hours</td><td>25%</td></tr><tr><td>24+ hours</td><td>50%</td></tr></table><p><strong>Note:</strong> SLA applies to business accounts only.</p>',
        '2024-01-01',
        'https://confluence.company.com/display/POLICY/Service+Level+Agreement',
        'POLICY-12348',
        CURRENT_TIMESTAMP(),
        array('sla', 'uptime', 'business', 'service-credits')
    )
""")

print("✅ Policy documentation table created with Confluence metadata")

# Create Vector Search index for policy docs
policy_index = f"{catalog}.{schema}.policy_docs_index"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, policy_index):
    print(f"Creating Vector Search index: {policy_index}")
    vsc.create_delta_sync_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=policy_index,
        source_table_name=f"{catalog}.{schema}.policy_docs",
        pipeline_type="TRIGGERED",
        primary_key="id",
        embedding_source_column='content',
        embedding_model_endpoint_name='databricks-gte-large-en'
    )
    wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, policy_index)
else:
    vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, policy_index).sync()

print(f"✅ Policy index ready: {policy_index}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Real-World Pattern: Confluence Integration
# MAGIC
# MAGIC This demonstrates a common enterprise pattern where documentation comes from Confluence:
# MAGIC
# MAGIC **Key Features:**
# MAGIC - `content`: Plain text version (used for vector search embeddings)
# MAGIC - `content_html`: Original HTML from Confluence API
# MAGIC - `source_url`: Link back to the original Confluence page
# MAGIC - `confluence_page_id`: Unique page identifier for API calls
# MAGIC - `tags`: Array of tags for filtering and categorization
# MAGIC - `last_updated`: Timestamp for freshness tracking
# MAGIC
# MAGIC **Integration Flow:**
# MAGIC 1. Confluence API fetches pages with metadata
# MAGIC 2. HTML content is parsed to plain text for embeddings
# MAGIC 3. Vector Search indexes the plain text
# MAGIC 4. Agent retrieves documents with source URLs
# MAGIC 5. Users get answers WITH citations to original docs
# MAGIC
# MAGIC **Benefits:**
# MAGIC - **Traceability**: Users can verify information at the source
# MAGIC - **Trust**: Citations increase confidence in AI responses
# MAGIC - **Compliance**: Audit trail for information provenance
# MAGIC - **Updates**: Links always point to latest version in Confluence

# COMMAND ----------

# DBTITLE 1,Test Vector Search Indexes
# Test all three indexes
print("Testing Product Docs Index:")
results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, product_docs_index).similarity_search(
    query_text="How do I restart my router?",
    columns=["product_name", "content"],
    num_results=1,
    disable_notice=True
)
print(results.get('result', {}).get('data_array', []))

print("\nTesting Troubleshooting Index:")
results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, troubleshooting_index).similarity_search(
    query_text="My wifi is not connecting",
    columns=["issue_type", "content"],
    num_results=1,
    disable_notice=True
)
print(results.get('result', {}).get('data_array', []))

print("\nTesting Policy Index (with source citations):")
results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, policy_index).similarity_search(
    query_text="What is your refund policy?",
    columns=["policy_type", "title", "content", "source_url", "confluence_page_id", "tags"],
    num_results=1,
    disable_notice=True
)
print(results.get('result', {}).get('data_array', []))
print("\nNote: Retrieved content includes source_url for citation!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Prompt Engineering
# MAGIC
# MAGIC We'll create different system prompts and compare their effectiveness.

# COMMAND ----------

# DBTITLE 1,Version 1: Basic System Prompt
system_prompt_v1 = """You are a customer support assistant. Help customers with their questions."""

print("System Prompt Version 1 (Basic):")
print(system_prompt_v1)

# COMMAND ----------

# DBTITLE 1,Version 2: Improved with Context
system_prompt_v2 = """You are a helpful customer support assistant for a telecommunications company.

Your capabilities:
- Look up customer information
- Check billing and payment details
- View support ticket history
- Access product documentation
- Provide troubleshooting assistance
- Explain company policies

Important: Only use information from your available tools and knowledge base. If you cannot find information, clearly state that you don't have that information available.

Call the appropriate tool to help the user with billing, support, or account info."""

print("\nSystem Prompt Version 2 (With Context):")
print(system_prompt_v2)

# COMMAND ----------

# DBTITLE 1,Version 3: Optimized with Guidelines (BEST)
system_prompt_v3 = """You are a professional customer support AI agent for a telecommunications and internet service provider.

Your role is to:
- Answer customer questions about their accounts, billing, and support tickets
- Provide technical troubleshooting guidance
- Explain company policies and procedures

CRITICAL KNOWLEDGE BASE RESTRICTION:
⚠️ YOU ARE STRICTLY FORBIDDEN FROM USING ANY INFORMATION NOT RETRIEVED FROM YOUR TOOLS ⚠️

STEP 1: ALWAYS use your tools first to search for information:
- If customer asks about a product → Use product_documentation_retriever
- If customer asks about troubleshooting → Use troubleshooting_guide_retriever  
- If customer asks about policies → Use policy_documentation_retriever
- If customer asks about account/billing → Use get_customer_info or get_billing_info

STEP 2: Check if tools returned relevant information:
- If tools return "No relevant documents found" or similar → YOU DO NOT HAVE THE ANSWER
- If tools return empty results → YOU DO NOT HAVE THE ANSWER
- If tools return documents that don't mention the specific product/issue → YOU DO NOT HAVE THE ANSWER

STEP 3: Response rules:
✅ IF tools found relevant info → Provide answer based ONLY on retrieved information
❌ IF tools found NO relevant info → Respond with:

"I don't have information about [specific thing] in our knowledge base. I cannot provide an answer to that question. Please contact our support team at [contact info] or check if you meant a different product model."

⚠️ ABSOLUTELY FORBIDDEN:
- Do NOT say "generally" or "typically" or "most routers"  
- Do NOT provide generic advice from your training data
- Do NOT use phrases like "I recommend" if not from knowledge base
- Do NOT make educated guesses
- Do NOT provide "standard procedures" not in the retrieved documents
- Do NOT say "If you need model-specific instructions..." - either you have it or you don't

IMPORTANT GUIDELINES:
- DO NOT mention internal tools or functions you are using
- DO NOT show your reasoning or intermediate steps
- Provide direct answers as if you have the information readily available
- Be professional, friendly, and concise
- If you need customer email to look up information, ask for it politely

MANDATORY CITATION REQUIREMENTS (CRITICAL):
- Retrieved policy documents include "[SOURCE URL: ...]" markers at the end
- You MUST extract and include these URLs in your final response
- Format: End your answer with "Source: [the URL from the retrieved document]"
- Example: "Source: https://confluence.company.com/display/POLICY/Billing+Dispute+Policy"
- If multiple documents are retrieved, list all source URLs
- This is REQUIRED for compliance and audit trail purposes

Remember: It's better to say "I don't know" than to provide unverified general information. Your reliability depends on only using your knowledge base."""

print("\nSystem Prompt Version 3 (Optimized with Citations):")
print(system_prompt_v3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Create Agent with LangGraph
# MAGIC
# MAGIC Now we'll create our agent using LangGraph, incorporating all the tools and knowledge bases.

# COMMAND ----------

# DBTITLE 1,Agent Implementation
import mlflow
from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
    VectorSearchRetrieverTool,
    DatabricksFunctionClient,
    set_uc_function_client,
)
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.pyfunc import ResponsesAgent
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint, DatabricksVectorSearchIndex
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from typing import Annotated, Any, Optional, Sequence, TypedDict, Union
import json
from uuid import uuid4

# Enable MLflow tracing
mlflow.langchain.autolog()
set_uc_function_client(DatabricksFunctionClient())

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]

def create_tool_calling_agent(
    model,
    tools,
    system_prompt: Optional[str] = None,
):
    model = model.bind_tools(tools)

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        return "continue" if isinstance(last, AIMessage) and last.tool_calls else "end"

    # Check if messages already contain a system prompt
    def prepend_system_if_needed(state):
        messages = state["messages"]
        # Check if first message is already a system message
        # LangChain messages have a 'type' attribute
        if messages and hasattr(messages[0], 'type') and messages[0].type == "system":
            # User provided system prompt, use it instead
            return messages
        elif system_prompt:
            # No system prompt in messages, prepend default
            return [{"role": "system", "content": system_prompt}] + messages
        else:
            return messages
    
    pre = RunnableLambda(prepend_system_if_needed)
    model_runnable = pre | model

    def call_model(state: AgentState, config: RunnableConfig):
        return {"messages": [model_runnable.invoke(state, config)]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", RunnableLambda(call_model))
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph.compile()

class CitationFormattingRetriever:
    """Wrapper for VectorSearchRetrieverTool that formats results with citations"""
    
    def __init__(self, index_name, name, description, num_results=2, columns=None):
        self.index_name = index_name
        self.name = name
        self.description = description
        self.num_results = num_results
        self.columns = columns or ["content"]
        self.vsc = VectorSearchClient(disable_notice=True)
        
    def _format_doc_with_citation(self, doc, columns):
        """Format a single document to include source URL in the text"""
        if isinstance(doc, list):
            # doc is array of values matching column order
            result_text = ""
            content_idx = None
            source_url_idx = None
            title_idx = None
            
            for idx, col in enumerate(columns):
                if col == "content":
                    content_idx = idx
                elif col == "source_url":
                    source_url_idx = idx
                elif col == "title":
                    title_idx = idx
            
            # Build formatted text
            if title_idx is not None and title_idx < len(doc):
                result_text += f"Document Title: {doc[title_idx]}\n\n"
            
            if content_idx is not None and content_idx < len(doc):
                result_text += f"{doc[content_idx]}\n\n"
            
            if source_url_idx is not None and source_url_idx < len(doc):
                result_text += f"[SOURCE URL: {doc[source_url_idx]}]"
            
            return result_text
        elif isinstance(doc, dict):
            result_text = ""
            if "title" in doc:
                result_text += f"Document Title: {doc['title']}\n\n"
            if "content" in doc:
                result_text += f"{doc['content']}\n\n"
            if "source_url" in doc:
                result_text += f"[SOURCE URL: {doc['source_url']}]"
            return result_text
        else:
            return str(doc)
    
    def invoke(self, query: str) -> str:
        """Retrieve documents and format with citations"""
        results = self.vsc.get_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
            index_name=self.index_name
        ).similarity_search(
            query_text=query,
            columns=self.columns,
            num_results=self.num_results,
            disable_notice=True
        )
        
        data_array = results.get('result', {}).get('data_array', [])
        
        if not data_array:
            return "No relevant documents found."
        
        formatted_docs = []
        for i, doc in enumerate(data_array):
            formatted_doc = self._format_doc_with_citation(doc, self.columns)
            formatted_docs.append(f"--- Document {i+1} ---\n{formatted_doc}")
        
        return "\n\n".join(formatted_docs)

class MultiRAGAgent(ResponsesAgent):
    def __init__(
        self,
        uc_tool_names: Sequence[str],
        llm_endpoint_name: str,
        system_prompt: str,
        retriever_configs: list[dict],
        max_history_messages: int = 20,
    ):
        self.llm_endpoint_name = llm_endpoint_name
        self.system_prompt = system_prompt
        self.max_history_messages = max_history_messages
        self.retriever_configs = retriever_configs

        # Initialize LLM
        self.llm = ChatDatabricks(endpoint=llm_endpoint_name)
        
        # Initialize UC Function tools
        self.tools: list[BaseTool] = UCFunctionToolkit(function_names=list(uc_tool_names)).tools

        # Add multiple retriever tools
        for config in retriever_configs:
            # Check if this is a policy retriever that needs citation formatting
            if "source_url" in config.get("columns", []):
                # Use custom citation-formatting retriever
                from langchain_core.tools import StructuredTool
                
                retriever = CitationFormattingRetriever(
                    index_name=config["index_name"],
                    name=config["tool_name"],
                    description=config["description"],
                    num_results=config.get("num_results", 2),
                    columns=config.get("columns")
                )
                
                tool = StructuredTool.from_function(
                    func=retriever.invoke,
                    name=config["tool_name"],
                    description=config["description"]
                )
                self.tools.append(tool)
            else:
                # Use standard retriever for other indexes
                tool_kwargs = {
                    "index_name": config["index_name"],
                    "name": config["tool_name"],
                    "description": config["description"],
                    "num_results": config.get("num_results", 2),
                }
                if "columns" in config:
                    tool_kwargs["columns"] = config["columns"]
                
                self.tools.append(VectorSearchRetrieverTool(**tool_kwargs))

        # Create agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, system_prompt)

    def _responses_to_cc(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        msg_type = message.get("type")
        if msg_type == "function_call":
            return [{
                "role": "assistant",
                "content": "tool_call",
                "tool_calls": [{
                    "id": message["call_id"],
                    "type": "function",
                    "function": {
                        "arguments": message["arguments"],
                        "name": message["name"],
                    },
                }],
            }]
        elif msg_type == "message" and isinstance(message["content"], list):
            return [{"role": message["role"], "content": content["text"]} for content in message["content"]]
        elif msg_type == "function_call_output":
            return [{
                "role": "tool",
                "content": message["output"],
                "tool_call_id": message["call_id"],
            }]
        filtered = {k: v for k, v in message.items() if k in {"role", "content", "name", "tool_calls", "tool_call_id"}}
        return [filtered] if filtered else []

    def _langchain_to_responses(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for message in messages:
            message = message.model_dump()
            if message["type"] == "ai":
                if tool_calls := message.get("tool_calls"):
                    return [
                        self.create_function_call_item(
                            id=message.get("id") or str(uuid4()),
                            call_id=tc["id"],
                            name=tc["name"],
                            arguments=json.dumps(tc["args"]),
                        )
                        for tc in tool_calls
                    ]
                # Safely extract content as string
                content = message.get("content", "")
                if isinstance(content, list):
                    # If content is a list, extract text from it
                    if content and isinstance(content[0], dict) and "text" in content[0]:
                        content = content[0]["text"]
                    else:
                        content = ""
                
                # Only create output if we have actual content
                if content:
                    mlflow.update_current_trace(response_preview=content)
                    return [self.create_text_output_item(
                        text=content,
                        id=message.get("id") or str(uuid4())
                    )]
            elif message["type"] == "tool":
                return [self.create_function_call_output_item(
                    call_id=message["tool_call_id"],
                    output=message["content"]
                )]
        return []

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs)

    def predict_stream(self, request: ResponsesAgentRequest):
        cc_msgs = []
        for msg in request.input:
            cc_msgs.extend(self._responses_to_cc(msg.model_dump()))

        if len(cc_msgs) > self.max_history_messages:
            cc_msgs = cc_msgs[-self.max_history_messages:]
            
        for event in self.agent.stream({"messages": cc_msgs}, stream_mode=["updates"]):
            if event[0] == "updates":
                for node_data in event[1].values():
                    for item in self._langchain_to_responses(node_data["messages"]):
                        yield mlflow.types.responses.ResponsesAgentStreamEvent(
                            type="response.output_item.done", item=item
                        )
                
    def get_resources(self):
        try:
            endpoint_name = self.llm.model
        except AttributeError:
            endpoint_name = self.llm.endpoint
        
        res = [DatabricksServingEndpoint(endpoint_name=endpoint_name)]
        for t in self.tools:
            if isinstance(t, VectorSearchRetrieverTool):
                res.extend(t.resources)
            elif hasattr(t, "uc_function_name"):
                res.append(DatabricksFunction(function_name=t.uc_function_name))

        # Add Vector Search indexes from retriever configs
        if hasattr(self, 'retriever_configs'):
            for config in self.retriever_configs:
                index_name = config.get("index_name")
                if index_name:
                    res.append(DatabricksVectorSearchIndex(index_name=index_name))
        
        return res

print("✅ Agent class defined")

# COMMAND ----------

# DBTITLE 1,Initialize Agent with Multiple Retrievers
# Configure multiple retrievers
retriever_configs = [
    {
        "index_name": product_docs_index,
        "tool_name": "product_documentation_retriever",
        "description": "Retrieves product documentation for routers, modems, and networking equipment. Use this for questions about product features, setup, and hardware specifications.",
        "num_results": 2,
        "vector_search_endpoint": VECTOR_SEARCH_ENDPOINT_NAME
    },
    {
        "index_name": troubleshooting_index,
        "tool_name": "troubleshooting_guide_retriever", 
        "description": "Retrieves technical troubleshooting guides and solutions. Use this for connectivity issues, error codes, and technical problems.",
        "num_results": 2,
        "vector_search_endpoint": VECTOR_SEARCH_ENDPOINT_NAME
    },
    {
        "index_name": policy_index,
        "tool_name": "policy_documentation_retriever",
        "description": "Retrieves company policies including billing, refunds, cancellations, SLA, and privacy policies. Returns formatted documents with [SOURCE URL: ...] markers. You MUST extract and cite these URLs in your response. Use this tool for any policy-related questions.",
        "num_results": 2,
        "columns": ["policy_type", "title", "content", "source_url", "confluence_page_id", "effective_date"],
        "vector_search_endpoint": VECTOR_SEARCH_ENDPOINT_NAME
    }
]

# Initialize agent with optimized prompt
agent = MultiRAGAgent(
    uc_tool_names=[f"{catalog}.{schema}.*"],
    llm_endpoint_name=LLM_ENDPOINT_NAME,
    system_prompt=system_prompt_v3,
    retriever_configs=retriever_configs,
    max_history_messages=20,
)

print("✅ Agent initialized with 3 vector search retrievers")
print(f"✅ Using LLM: {LLM_ENDPOINT_NAME}")
print(f"✅ Total tools available: {len(agent.tools)}")

# COMMAND ----------

# DBTITLE 1,Verify Custom Citation Retriever Works
print("Testing custom citation-formatting retriever...")
print("=" * 80)

# Test the custom retriever directly
test_retriever = CitationFormattingRetriever(
    index_name=policy_index,
    name="test_policy_retriever",
    description="Test retriever",
    num_results=1,
    columns=["policy_type", "title", "content", "source_url"]
)

test_result = test_retriever.invoke("refund policy")
print("Custom Retriever Output:")
print(test_result)
print("\n" + "=" * 80)

if "[SOURCE URL:" in test_result and "confluence.company.com" in test_result:
    print("✅ SUCCESS: Custom retriever properly formats documents with source URLs!")
else:
    print("⚠️  WARNING: Source URL not found in formatted output")
    print("   Check that policy documents have source_url populated")

print("=" * 80)

# COMMAND ----------

# DBTITLE 1,Test Agent with Different Queries
from mlflow.types.responses import ResponsesAgentRequest

def extract_response_text(response):
    """Helper function to safely extract text from agent response"""
    try:
        if hasattr(response.output[-1], 'content'):
            content = response.output[-1].content
            if isinstance(content, list) and len(content) > 0:
                last_content = content[-1]
                if hasattr(last_content, 'text'):
                    return last_content.text
                elif isinstance(last_content, dict) and 'text' in last_content:
                    return last_content['text']
                else:
                    return str(last_content)
            else:
                return str(content)
        else:
            return str(response.output[-1])
    except Exception as e:
        return f"Error extracting response: {e}\nFull response: {response}"

# Test 1: Customer lookup
print("=" * 80)
print("TEST 1: Customer Information Lookup")
print("=" * 80)
request = ResponsesAgentRequest(input=[{"role": "user", "content": "Give me information about john21@example.net"}])
response = agent.predict(request)
print(extract_response_text(response))

print("\n" + "=" * 80)
print("TEST 2: Product Documentation (RAG)")
print("=" * 80)
request = ResponsesAgentRequest(input=[{"role": "user", "content": "How do I restart my ADSL-R500 router if it shows Error Code 01?"}])
response = agent.predict(request)
print(extract_response_text(response))

print("\n" + "=" * 80)
print("TEST 3: Policy Information with Source Citation (RAG)")
print("=" * 80)
request = ResponsesAgentRequest(input=[{"role": "user", "content": "What is your refund policy if I cancel my service?"}])
response = agent.predict(request)
response_text = extract_response_text(response)
print(response_text)
print("\n⚠️ Note: Check if the response includes the Confluence source URL!")

print("\n" + "=" * 80)
print("TEST 4: Troubleshooting (RAG)")
print("=" * 80)
request = ResponsesAgentRequest(input=[{"role": "user", "content": "My internet is very slow, what should I do?"}])
response = agent.predict(request)
print(extract_response_text(response))

print("\n" + "=" * 80)
print("TEST 5: SLA Policy with Citation (RAG)")
print("=" * 80)
request = ResponsesAgentRequest(input=[{"role": "user", "content": "What happens if my internet goes down for 10 hours? Do I get any compensation?"}])
response = agent.predict(request)
response_text = extract_response_text(response)
print(response_text)
print("\n⚠️ Note: Response should cite the SLA policy Confluence URL!")

# COMMAND ----------

print("\n" + "=" * 80)
print("TEST 6: I Don't Know")
print("=" * 80)
request = ResponsesAgentRequest(input=[{"role": "user", "content": "How do I restart my H500 router"}])
response = agent.predict(request)
response_text = extract_response_text(response)
print(response_text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Agent Evaluation with MLflow
# MAGIC
# MAGIC Agent evaluation is critical for measuring and improving agent performance over time.
# MAGIC
# MAGIC MLflow provides comprehensive evaluation capabilities:
# MAGIC - Create evaluation datasets with expected outputs
# MAGIC - Define custom scorers and guidelines
# MAGIC - Run evaluations across different model versions
# MAGIC - Compare results and track improvements
# MAGIC - Built-in metrics: Groundedness, Relevance, Safety

# COMMAND ----------

# DBTITLE 1,Setup MLflow Experiment
import mlflow

experiment_name = f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/ai_agent_demo"
mlflow.set_experiment(experiment_name)

print(f"✅ MLflow experiment: {experiment_name}")

# COMMAND ----------

# DBTITLE 1,Create Evaluation Dataset
from pyspark.sql.functions import monotonically_increasing_id, pandas_udf, concat_ws, lit, col, from_json, struct
from pyspark.sql.types import StringType, ArrayType
import pandas as pd

print("Creating evaluation dataset...")

df = spark.table(f"{catalog}.{schema}.customers").limit(10).withColumn("row_id", monotonically_increasing_id())

@pandas_udf(StringType())
def generate_question(email: pd.Series, row_id: pd.Series) -> pd.Series:
    templates = [
        "What is the phone number of {email}?",
        "What is the current account status for {email}?",
        "What city does {email} live in?",
        "What is the full address of {email}?",
        "Show me the billing information for {email}.",
        "Are there any support tickets for {email}?",
        "What is your refund policy?",
        "How do I restart my router if it shows Error Code 01?",
        "My internet is slow, what should I do?",
        "What happens if my service is down for extended periods?",
    ]
    return pd.Series([
        templates[int(i) % len(templates)].format(email=e)
        for i, e in zip(row_id, email)
    ])

df = df.withColumn("question", generate_question("email", "row_id"))

df_pd = df.toPandas()
df_clean = spark.createDataFrame(df_pd)

df_clean = df_clean.withColumn(
    "prompt",
    concat_ws(
        " ",
        lit("You are evaluating an AI customer support system."),
        lit("Based on the following customer record:"),
        concat_ws(", ",
            df_clean.name, df_clean.email, df_clean.phone,
            df_clean.address, df_clean.city, df_clean.state, df_clean.zip_code,
            df_clean.account_status
        ),
        lit("Generate a JSON array of 2-3 factual statements (expected_facts) that should be included in the correct answer. Return only a valid JSON array of strings."),
        lit("Question:"), df_clean.question
    )
)

df_clean.createOrReplaceTempView("eval_questions")

final_df_raw = spark.sql(f"""
SELECT 
  question,
  AI_QUERY('{LLM_ENDPOINT_NAME}', prompt) AS expected_facts_json
FROM eval_questions
""")

final_df = final_df_raw.withColumn(
    "expected_facts",
    from_json(col("expected_facts_json"), ArrayType(StringType()))
)

eval_df = final_df.withColumn("inputs", struct("question")) \
                  .withColumn("predictions", lit("")) \
                  .withColumn("expectations", struct("expected_facts")) \
                  .select("inputs", "predictions", "expectations")

print("✅ Evaluation dataset created")
display(eval_df.limit(5))

# COMMAND ----------

# DBTITLE 1,Save and Register MLflow Evaluation Dataset
import mlflow.genai.datasets

eval_dataset_table_name = f"{catalog}.{schema}.ai_agent_eval_dataset"

try:
    mlflow_eval_dataset = mlflow.genai.datasets.get_dataset(eval_dataset_table_name)
    print(f"✅ Using existing evaluation dataset: {eval_dataset_table_name}")
except Exception as e:
    if 'does not exist' in str(e):
        mlflow_eval_dataset = mlflow.genai.datasets.create_dataset(eval_dataset_table_name)
        mlflow_eval_dataset.merge_records(eval_df)
        print(f"✅ Created new evaluation dataset: {eval_dataset_table_name}")

display(mlflow_eval_dataset.to_df().head(10))

# COMMAND ----------

# DBTITLE 1,Define Custom Scorers
from mlflow.genai.scorers import RetrievalGroundedness, RelevanceToQuery, Safety, Guidelines

scorers = [
    RetrievalGroundedness(),
    RelevanceToQuery(),
    Safety(),
    Guidelines(
        guidelines="""
        Response must be professional and not show internal reasoning or tools.
        - Do NOT mention that you are looking up information
        - Do NOT mention tools, functions, or databases being used
        - Do NOT show intermediate steps or reasoning
        - Provide direct, helpful answers
        - For policy questions, MUST include source URLs when available
        """,
        name="professional_response_without_tools",
    )
]

print("✅ Defined evaluation scorers:")
for scorer in scorers:
    print(f"  - {scorer.__class__.__name__}")

# COMMAND ----------

# DBTITLE 1,Create Prediction Wrapper for Evaluation
def predict_for_eval(question):
    try:
        request = ResponsesAgentRequest(input=[{"role": "user", "content": question}])
        response = agent.predict(request)
        return extract_response_text(response)
    except Exception as e:
        return f"Error: {str(e)}"

test_question = "What is your refund policy?"
test_answer = predict_for_eval(test_question)
print(f"Test Question: {test_question}")
print(f"Test Answer: {test_answer}")

# COMMAND ----------

# DBTITLE 1,Run Agent Evaluation
print("=" * 80)
print("Running agent evaluation...")
print("=" * 80)

with mlflow.start_run(run_name="multi_rag_agent_evaluation_v1") as eval_run:
    eval_results = mlflow.genai.evaluate(
        data=mlflow_eval_dataset,
        predict_fn=predict_for_eval,
        scorers=scorers
    )

print("\n✅ Evaluation complete!")
print(f"Run ID: {eval_run.info.run_id}")

print("\n📊 Evaluation Metrics Summary:")
metrics_df = eval_results.metrics
for metric_name, metric_value in metrics_df.items():
    if isinstance(metric_value, (int, float)):
        print(f"  {metric_name}: {metric_value:.3f}")

# COMMAND ----------

# DBTITLE 1,View Detailed Evaluation Results
print("Detailed evaluation results:")
# Access the underlying DataFrame directly
import pandas as pd
if hasattr(eval_results, 'tables') and isinstance(eval_results.tables, dict):
    display(eval_results.tables['eval_results_table'])
else:
    # Just show metrics if tables aren't available
    print("Evaluation Metrics:")
    print(eval_results.metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compare with Different System Prompt
# MAGIC
# MAGIC Let's evaluate with a different system prompt to see the impact on scores.

# COMMAND ----------

# DBTITLE 1,Test Alternative System Prompt
alternative_system_prompt = """You are a customer support assistant. 
Help customers by looking up their information in our database.
Always be helpful and professional."""

agent_alternative = MultiRAGAgent(
    uc_tool_names=[f"{catalog}.{schema}.*"],
    llm_endpoint_name=LLM_ENDPOINT_NAME,
    system_prompt=alternative_system_prompt,
    retriever_configs=retriever_configs,
    max_history_messages=20,
)

def predict_for_eval_alternative(question):
    try:
        request = ResponsesAgentRequest(input=[{"role": "user", "content": question}])
        response = agent_alternative.predict(request)
        return extract_response_text(response)
    except Exception as e:
        return f"Error: {str(e)}"

with mlflow.start_run(run_name="multi_rag_agent_evaluation_v2_simple_prompt") as eval_run_v2:
    eval_results_v2 = mlflow.genai.evaluate(
        data=mlflow_eval_dataset,
        predict_fn=predict_for_eval_alternative,
        scorers=scorers
    )

print("\n✅ Alternative prompt evaluation complete!")
print(f"Run ID: {eval_run_v2.info.run_id}")

print("\n📊 Alternative Prompt Metrics Summary:")
for metric_name, metric_value in eval_results_v2.metrics.items():
    if isinstance(metric_value, (int, float)):
        print(f"  {metric_name}: {metric_value:.3f}")

# COMMAND ----------

# DBTITLE 1,Compare Evaluation Results
print("=" * 80)
print("EVALUATION COMPARISON")
print("=" * 80)

comparison_data = []
for metric_name in eval_results.metrics:
    if isinstance(eval_results.metrics[metric_name], (int, float)):
        v1_score = eval_results.metrics[metric_name]
        v2_score = eval_results_v2.metrics.get(metric_name, 0)
        diff = v1_score - v2_score
        comparison_data.append({
            "Metric": metric_name,
            "Optimized Prompt (v3)": f"{v1_score:.3f}",
            "Simple Prompt": f"{v2_score:.3f}",
            "Difference": f"{diff:+.3f}"
        })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

print("\n💡 Tip: Open the MLflow Experiment UI to compare runs visually!")
print(f"   Experiment: {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Log and Register Agent with MLflow

# COMMAND ----------

# DBTITLE 1,Create Agent Configuration File
import yaml
import os

# Create agent configuration
agent_config = {
    "config_version_name": "multi_rag_claude_sonnet",
    "uc_tool_names": [f"{catalog}.{schema}.*"],
    "llm_endpoint_name": LLM_ENDPOINT_NAME,
    "system_prompt": system_prompt_v3,
    "retriever_configs": retriever_configs,
    "max_history_messages": 20,
}

# Save configuration
config_path = os.path.abspath(os.path.join(os.getcwd())) + "/agent_config.yaml"
with open(config_path, 'w') as f:
    yaml.dump(agent_config, f)

print("✅ Agent configuration saved")
print(yaml.dump(agent_config, default_flow_style=False))

# COMMAND ----------

# DBTITLE 1,Create Agent Python Module
agent_code = '''
import json
from typing import Annotated, Any, Generator, Optional, Sequence, TypedDict, Union
from uuid import uuid4

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
    VectorSearchRetrieverTool,
    DatabricksFunctionClient,
    set_uc_function_client,
)
from databricks.vector_search.client import VectorSearchClient

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool, StructuredTool

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode

from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.models import ModelConfig
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint, DatabricksVectorSearchIndex
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

mlflow.langchain.autolog()
set_uc_function_client(DatabricksFunctionClient())

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]

def create_tool_calling_agent(model, tools, system_prompt: Optional[str] = None):
    model = model.bind_tools(tools)

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        return "continue" if isinstance(last, AIMessage) and last.tool_calls else "end"

    # Check if messages already contain a system prompt
    def prepend_system_if_needed(state):
        messages = state["messages"]
        # Check if first message is already a system message
        # LangChain messages have a 'type' attribute
        if messages and hasattr(messages[0], 'type') and messages[0].type == "system":
            # User provided system prompt, use it instead
            return messages
        elif system_prompt:
            # No system prompt in messages, prepend default
            return [{"role": "system", "content": system_prompt}] + messages
        else:
            return messages
    
    pre = RunnableLambda(prepend_system_if_needed)
    model_runnable = pre | model

    def call_model(state: AgentState, config: RunnableConfig):
        return {"messages": [model_runnable.invoke(state, config)]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", RunnableLambda(call_model))
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph.compile()

class CitationFormattingRetriever:
    """Wrapper for VectorSearchRetrieverTool that formats results with citations"""
    
    def __init__(self, index_name, name, description, num_results=2, columns=None, vector_search_endpoint=None):
        self.index_name = index_name
        self.name = name
        self.description = description
        self.num_results = num_results
        self.columns = columns or ["content"]
        self.vector_search_endpoint = vector_search_endpoint
        self.vsc = VectorSearchClient(disable_notice=True)
        
    def _format_doc_with_citation(self, doc, columns):
        """Format a single document to include source URL in the text"""
        if isinstance(doc, list):
            result_text = ""
            content_idx = None
            source_url_idx = None
            title_idx = None
            
            for idx, col in enumerate(columns):
                if col == "content":
                    content_idx = idx
                elif col == "source_url":
                    source_url_idx = idx
                elif col == "title":
                    title_idx = idx
            
            if title_idx is not None and title_idx < len(doc):
                result_text += f"Document Title: {doc[title_idx]}\\n\\n"
            
            if content_idx is not None and content_idx < len(doc):
                result_text += f"{doc[content_idx]}\\n\\n"
            
            if source_url_idx is not None and source_url_idx < len(doc):
                result_text += f"[SOURCE URL: {doc[source_url_idx]}]"
            
            return result_text
        elif isinstance(doc, dict):
            result_text = ""
            if "title" in doc:
                result_text += f"Document Title: {doc['title']}\\n\\n"
            if "content" in doc:
                result_text += f"{doc['content']}\\n\\n"
            if "source_url" in doc:
                result_text += f"[SOURCE URL: {doc['source_url']}]"
            return result_text
        else:
            return str(doc)
    
    def invoke(self, query: str) -> str:
        """Retrieve documents and format with citations"""
        results = self.vsc.get_index(
            endpoint_name=self.vector_search_endpoint,
            index_name=self.index_name
        ).similarity_search(
            query_text=query,
            columns=self.columns,
            num_results=self.num_results,
            disable_notice=True
        )
        
        data_array = results.get('result', {}).get('data_array', [])
        
        if not data_array:
            return "No relevant documents found."
        
        formatted_docs = []
        for i, doc in enumerate(data_array):
            formatted_doc = self._format_doc_with_citation(doc, self.columns)
            formatted_docs.append(f"--- Document {i+1} ---\\n{formatted_doc}")
        
        return "\\n\\n".join(formatted_docs)

class MultiRAGAgent(ResponsesAgent):
    def __init__(
        self,
        uc_tool_names: Sequence[str],
        llm_endpoint_name: str,
        system_prompt: Optional[str] = None,
        retriever_configs: Optional[list[dict]] = None,
        max_history_messages: int = 20,
    ):
        self.llm_endpoint_name = llm_endpoint_name
        self.system_prompt = system_prompt
        self.max_history_messages = max_history_messages
        self.retriever_configs = retriever_configs

        self.llm = ChatDatabricks(endpoint=llm_endpoint_name)
        self.tools: list[BaseTool] = UCFunctionToolkit(function_names=list(uc_tool_names)).tools

        if retriever_configs:
            for config in retriever_configs:
                if "source_url" in config.get("columns", []):
                    retriever = CitationFormattingRetriever(
                        index_name=config["index_name"],
                        name=config["tool_name"],
                        description=config["description"],
                        num_results=config.get("num_results", 2),
                        columns=config.get("columns"),
                        vector_search_endpoint=config.get("vector_search_endpoint")
                    )
                    
                    tool = StructuredTool.from_function(
                        func=retriever.invoke,
                        name=config["tool_name"],
                        description=config["description"]
                    )
                    self.tools.append(tool)
                else:
                    tool_kwargs = {
                        "index_name": config["index_name"],
                        "name": config["tool_name"],
                        "description": config["description"],
                        "num_results": config.get("num_results", 2),
                    }
                    if "columns" in config:
                        tool_kwargs["columns"] = config["columns"]
                    
                    self.tools.append(VectorSearchRetrieverTool(**tool_kwargs))

        self.agent = create_tool_calling_agent(self.llm, self.tools, system_prompt)

    def _responses_to_cc(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        msg_type = message.get("type")
        if msg_type == "function_call":
            return [{
                "role": "assistant",
                "content": "tool_call",
                "tool_calls": [{
                    "id": message["call_id"],
                    "type": "function",
                    "function": {
                        "arguments": message["arguments"],
                        "name": message["name"],
                    },
                }],
            }]
        elif msg_type == "message" and isinstance(message["content"], list):
            return [{"role": message["role"], "content": content["text"]} for content in message["content"]]
        elif msg_type == "function_call_output":
            return [{
                "role": "tool",
                "content": message["output"],
                "tool_call_id": message["call_id"],
            }]
        filtered = {k: v for k, v in message.items() if k in {"role", "content", "name", "tool_calls", "tool_call_id"}}
        return [filtered] if filtered else []

    def _langchain_to_responses(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for message in messages:
            message = message.model_dump()
            if message["type"] == "ai":
                if tool_calls := message.get("tool_calls"):
                    return [
                        self.create_function_call_item(
                            id=message.get("id") or str(uuid4()),
                            call_id=tc["id"],
                            name=tc["name"],
                            arguments=json.dumps(tc["args"]),
                        )
                        for tc in tool_calls
                    ]
                # Safely extract content as string
                content = message.get("content", "")
                if isinstance(content, list):
                    # If content is a list, extract text from it
                    if content and isinstance(content[0], dict) and "text" in content[0]:
                        content = content[0]["text"]
                    else:
                        content = ""
                
                # Only create output if we have actual content
                if content:
                    mlflow.update_current_trace(response_preview=content)
                    return [self.create_text_output_item(
                        text=content,
                        id=message.get("id") or str(uuid4())
                    )]
            elif message["type"] == "tool":
                return [self.create_function_call_output_item(
                    call_id=message["tool_call_id"],
                    output=message["content"]
                )]
        return []

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self, request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        cc_msgs = []
        mlflow.update_current_trace(request_preview=request.input[0].content)
        for msg in request.input:
            cc_msgs.extend(self._responses_to_cc(msg.model_dump()))

        if len(cc_msgs) > self.max_history_messages:
            cc_msgs = cc_msgs[-self.max_history_messages:]
        for event in self.agent.stream({"messages": cc_msgs}, stream_mode=["updates", "messages"]):
            if event[0] == "updates":
                for node_data in event[1].values():
                    for item in self._langchain_to_responses(node_data["messages"]):
                        yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)
            elif event[0] == "messages":
                try:
                    chunk = event[1][0]
                    if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                        yield ResponsesAgentStreamEvent(
                            **self.create_text_delta(delta=content, item_id=chunk.id),
                        )
                except Exception:
                    pass
                
    def get_resources(self):
        try:
            endpoint_name = self.llm.model
        except AttributeError:
            endpoint_name = self.llm.endpoint
        
        res = [DatabricksServingEndpoint(endpoint_name=endpoint_name)]
        for t in self.tools:
            if isinstance(t, VectorSearchRetrieverTool):
                res.extend(t.resources)
            elif hasattr(t, "uc_function_name"):
                res.append(DatabricksFunction(function_name=t.uc_function_name))

        # Add Vector Search indexes from retriever configs
        if hasattr(self, 'retriever_configs') and self.retriever_configs:
            for config in self.retriever_configs:
                index_name = config.get("index_name")
                if index_name:
                    res.append(DatabricksVectorSearchIndex(index_name=index_name))
        
        return res

model_config = ModelConfig(development_config="agent_config.yaml")

AGENT = MultiRAGAgent(
    uc_tool_names=model_config.get("uc_tool_names"),
    llm_endpoint_name=model_config.get("llm_endpoint_name"),
    system_prompt=model_config.get("system_prompt"),
    retriever_configs=model_config.get("retriever_configs"),
    max_history_messages=model_config.get("max_history_messages"),
)

mlflow.models.set_model(AGENT)
'''

# Save agent code
agent_code_path = os.path.abspath(os.path.join(os.getcwd())) + "/agent.py"
with open(agent_code_path, 'w') as f:
    f.write(agent_code)

print("✅ Agent code saved")

# COMMAND ----------

# DBTITLE 1,Log Agent to MLflow
# Log model
with mlflow.start_run(run_name="multi_rag_claude_sonnet_agent") as run:
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model=agent_code_path,
        model_config=config_path,
        input_example={"input": [{"role": "user", "content": "What is your refund policy?"}]},
        resources=agent.get_resources(),
        extra_pip_requirements=["databricks-connect"]
    )

print(f"✅ Agent logged to MLflow")
print(f"Run ID: {run.info.run_id}")
print(f"Model URI: {logged_agent_info.model_uri}")

# COMMAND ----------

# DBTITLE 1,Register Agent to Unity Catalog
from mlflow import MlflowClient

UC_MODEL_NAME = f"{catalog}.{schema}.{MODEL_NAME}"

client = MlflowClient()
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri,
    name=UC_MODEL_NAME,
    tags={"model": "multi_rag_agent", "llm": "claude_sonnet_4_5"}
)

client.set_registered_model_alias(
    name=UC_MODEL_NAME,
    alias="production",
    version=uc_registered_model_info.version
)

print(f"✅ Agent registered to Unity Catalog: {UC_MODEL_NAME}")
print(f"Version: {uc_registered_model_info.version}")
displayHTML(f'<a href="/explore/data/models/{catalog}/{schema}/{MODEL_NAME}" target="_blank">View Model in Unity Catalog</a>')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Deploy Agent to Model Serving Endpoint

# COMMAND ----------

# DBTITLE 1,Deploy to Model Serving
from databricks import agents
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
import time

w = WorkspaceClient()

# Check if endpoint exists and deploy
try:
    endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
    print(f"Endpoint {ENDPOINT_NAME} already exists. Updating...")
    
    # Update endpoint with new model version
    agents.deploy(
        UC_MODEL_NAME,
        uc_registered_model_info.version,
        endpoint_name=ENDPOINT_NAME,
        tags={"project": "ai_agent_demo", "llm": "claude_sonnet_4_5"}
    )
except Exception as e:
    if "does not exist" in str(e):
        print(f"Creating new endpoint: {ENDPOINT_NAME}")
        agents.deploy(
            UC_MODEL_NAME,
            uc_registered_model_info.version,
            endpoint_name=ENDPOINT_NAME,
            tags={"project": "ai_agent_demo", "llm": "claude_sonnet_4_5"}
        )
    else:
        raise e

# Wait for endpoint to be ready
print("Waiting for endpoint to be ready...")
for i in range(200):
    state = w.serving_endpoints.get(ENDPOINT_NAME).state
    if state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
        if i % 2 == 0:
            print(f"Endpoint deploying... ({i * 10}s)")
        time.sleep(10)
    elif state.ready == EndpointStateReady.READY:
        print('✅ Endpoint is ready!')
        break
    else:
        print(f"Endpoint state: {state}")
        time.sleep(10)

displayHTML(f'<a href="/ml/endpoints/{ENDPOINT_NAME}" target="_blank">View Model Serving Endpoint</a>')

print("\n✅ Agent deployed successfully!")
print(f"Resources declared in agent:")
resources = agent.get_resources()
for resource in resources:
    print(f"  - {resource}")

# Verify that Vector Search indexes are properly declared
vs_indexes = [r for r in resources if "VectorSearchIndex" in str(type(r))]
print(f"\nVector Search Indexes detected: {len(vs_indexes)}")
for vs_idx in vs_indexes:
    print(f"  - {vs_idx.name}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Invoke Agent with Dynamic System Prompts
# MAGIC
# MAGIC This section demonstrates how to call the deployed agent with different system prompts at runtime.

# COMMAND ----------

# DBTITLE 1,Helper Function: Invoke Agent with Custom Prompt
import requests
import json

def invoke_agent_with_custom_prompt(query, system_prompt_override=None):
    """
    Invoke the deployed agent with an optional custom system prompt.
    
    Args:
        query: The user query
        system_prompt_override: Optional custom system prompt to use instead of the default
    """
    # Get workspace URL and token
    workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    
    # Prepare the request
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Add custom system prompt if provided
    if system_prompt_override is not None:
        payload = {
            "input": [{"role": "system", "content": system_prompt_override}]
        }

        payload["input"].append({"role": "user", "content": query})
    
    else:
        payload = {"input": [{"role": "user", "content": query}],
            "extra_body": {}
        }
    
    # Call the endpoint
    url = f"{workspace_url}/serving-endpoints/{ENDPOINT_NAME}/invocations"
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error calling endpoint: {response.status_code} - {response.text}")

print("✅ Helper function defined")

# COMMAND ----------

# DBTITLE 1,Test 1: Default System Prompt
print("=" * 80)
print("TEST 1: Using Default System Prompt")
print("=" * 80)

query = "How do I restart my ADSL-R500 router?"

result = invoke_agent_with_custom_prompt(query)
print(f"\nQuery: {query}")
print(f"\nResponse: {result}")

# COMMAND ----------

# DBTITLE 1,Test 2: Friendly/Casual Tone
print("=" * 80)
print("TEST 2: Custom Prompt - Friendly & Casual Tone")
print("=" * 80)

friendly_prompt = """You are a friendly and casual customer support buddy! 

Be super helpful and use a warm, conversational tone. Add empathy and understanding.
Use the tools to look up information, but present it in a friendly, natural way.

Remember:
- Be warm and empathetic
- Use conversational language
- Show you understand their frustration
- Don't mention internal tools or functions
- Only use information from your available tools and knowledge base
- If you don't have information, be honest and say you don't have that information"""

query = "My wifi keeps disconnecting and it's driving me crazy!"

result = invoke_agent_with_custom_prompt(query, friendly_prompt)
print(f"\nQuery: {query}")
print(f"\nSystem Prompt: {friendly_prompt}")
print(f"\nResponse: {result}")

# COMMAND ----------

# DBTITLE 1,Test 3: Technical/Detailed Response
print("=" * 80)
print("TEST 3: Custom Prompt - Technical & Detailed")
print("=" * 80)

technical_prompt = """You are a highly technical support specialist for advanced users.

Provide detailed, technical information including:
- Specific technical steps
- System diagnostics
- Technical terminology
- Root cause analysis
- Advanced troubleshooting options

Use the tools to gather information and provide comprehensive technical guidance.

Important: Only provide information from your knowledge base. If you don't have specific technical details, clearly state that."""

query = "What could cause Error Code 01 on my router and how do I diagnose it?"

result = invoke_agent_with_custom_prompt(query, technical_prompt)
print(f"\nQuery: {query}")
print(f"\nSystem Prompt: {technical_prompt}")
print(f"\nResponse: {result}")

# COMMAND ----------

# DBTITLE 1,Test 4: Concise/Brief Responses
print("=" * 80)
print("TEST 4: Custom Prompt - Brief & Concise")
print("=" * 80)

brief_prompt = """You are a customer support agent who provides BRIEF, CONCISE answers.

Rules:
- Keep responses under 3 sentences when possible
- Get straight to the point
- No unnecessary explanations
- Use bullet points for steps
- Be efficient with words
- Only use information from your knowledge base
- If you don't have information, simply say "I don't have that information"""

query = "What's your refund policy?"

result = invoke_agent_with_custom_prompt(query, brief_prompt)
print(f"\nQuery: {query}")
print(f"\nSystem Prompt: {brief_prompt}")
print(f"\nResponse: {result}")

# COMMAND ----------

# DBTITLE 1,Test 5: Multilingual Support (Spanish)
print("=" * 80)
print("TEST 5: Custom Prompt - Spanish Language")
print("=" * 80)

spanish_prompt = """Eres un asistente de soporte al cliente profesional que habla español.

Proporciona respuestas útiles en español. Usa las herramientas disponibles para buscar información.

Reglas:
- Responde SIEMPRE en español
- Sé profesional y cortés
- No menciones herramientas internas
- Proporciona respuestas claras y útiles
- Solo usa información de tus herramientas y base de conocimientos
- Si no tienes información, di claramente "No tengo esa información disponible\""""

query = "¿Cuál es la política de reembolso?"

result = invoke_agent_with_custom_prompt(query, spanish_prompt)
print(f"\nQuery: {query}")
print(f"\nSystem Prompt: {spanish_prompt}")
print(f"\nResponse: {result}")

# COMMAND ----------

print("=" * 80)
print("TEST 6: Custom Prompt - Stupid")
print("=" * 80)

stupid_prompt = """It doesn't matter what the question is, just answer something totally irrelevant from the question"""

query = "My wifi keeps disconnecting and it's driving me crazy!"

result = invoke_agent_with_custom_prompt(query, stupid_prompt)
print(f"\nQuery: {query}")
print(f"\nSystem Prompt: {stupid_prompt}")
print(f"\nResponse: {result}")

# COMMAND ----------

print("=" * 80)
print("TEST 7: Using Default System Prompt - I Don't Know")
print("=" * 80)

query = "How do I restart my H500 router?"

result = invoke_agent_with_custom_prompt(query)
print(f"\nQuery: {query}")
print(f"\nResponse: {result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Microsoft Teams Integration
# MAGIC
# MAGIC Send agent responses to Microsoft Teams channels using incoming webhooks.

# COMMAND ----------

# DBTITLE 1,Microsoft Teams Integration Class
import pymsteams
import json
from datetime import datetime

class TeamsIntegration:
    """
    Integration with Microsoft Teams to send agent responses.
    """
    
    def __init__(self, webhook_url):
        """
        Initialize Teams integration with webhook URL.
        
        Args:
            webhook_url: The incoming webhook URL for your Teams channel
        """
        self.webhook_url = webhook_url
    
    def send_simple_message(self, message, title="AI Agent Response"):
        """
        Send a simple text message to Teams.
        
        Args:
            message: The message text
            title: Optional title for the message card
        """
        teams_message = pymsteams.connectorcard(self.webhook_url)
        teams_message.title(title)
        teams_message.text(message)
        teams_message.send()
    
    def send_agent_response(self, query, response, metadata=None):
        """
        Send a formatted agent response to Teams.
        
        Args:
            query: The original user query
            response: The agent's response
            metadata: Optional metadata (model name, latency, etc.)
        """
        teams_message = pymsteams.connectorcard(self.webhook_url)
        
        # Set title and color
        teams_message.title("🤖 AI Agent Response")
        teams_message.color("0078D4")  # Microsoft blue
        
        # Add query section
        query_section = pymsteams.cardsection()
        query_section.activityTitle("User Query")
        query_section.activityText(query)
        teams_message.addSection(query_section)
        
        # Add response section
        response_section = pymsteams.cardsection()
        response_section.activityTitle("Agent Response")
        response_section.activityText(response)
        teams_message.addSection(response_section)
        
        # Add metadata if provided
        if metadata:
            metadata_section = pymsteams.cardsection()
            metadata_section.activityTitle("Response Details")
            
            for key, value in metadata.items():
                metadata_section.addFact(key, str(value))
            
            teams_message.addSection(metadata_section)
        
        # Add timestamp
        teams_message.addSection(self._create_timestamp_section())
        
        teams_message.send()
    
    def send_knowledge_base_response(self, query, response, sources):
        """
        Send agent response with knowledge base sources to Teams.
        
        Args:
            query: The user query
            response: The agent's response
            sources: List of knowledge base sources used
        """
        teams_message = pymsteams.connectorcard(self.webhook_url)
        
        teams_message.title("🤖 AI Agent with RAG Response")
        teams_message.color("00B294")  # Green for knowledge-enhanced
        
        # Query section
        query_section = pymsteams.cardsection()
        query_section.activityTitle("📝 User Query")
        query_section.activityText(query)
        teams_message.addSection(query_section)
        
        # Response section
        response_section = pymsteams.cardsection()
        response_section.activityTitle("💬 AI Response")
        response_section.activityText(response)
        teams_message.addSection(response_section)
        
        # Sources section
        if sources:
            sources_section = pymsteams.cardsection()
            sources_section.activityTitle("📚 Knowledge Sources Used")
            sources_text = "\\n\\n".join([f"• {source}" for source in sources])
            sources_section.activityText(sources_text)
            teams_message.addSection(sources_section)
        
        # Timestamp
        teams_message.addSection(self._create_timestamp_section())
        
        teams_message.send()
    
    def send_error_notification(self, error_message, context=None):
        """
        Send error notification to Teams.
        
        Args:
            error_message: The error message
            context: Optional context about the error
        """
        teams_message = pymsteams.connectorcard(self.webhook_url)
        
        teams_message.title("⚠️ AI Agent Error")
        teams_message.color("D13438")  # Red for errors
        
        error_section = pymsteams.cardsection()
        error_section.activityTitle("Error Details")
        error_section.activityText(error_message)
        
        if context:
            for key, value in context.items():
                error_section.addFact(key, str(value))
        
        teams_message.addSection(error_section)
        teams_message.addSection(self._create_timestamp_section())
        
        teams_message.send()
    
    def _create_timestamp_section(self):
        """Create a section with timestamp."""
        timestamp_section = pymsteams.cardsection()
        timestamp_section.addFact("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        return timestamp_section

print("✅ Teams Integration class defined")

# COMMAND ----------

# DBTITLE 1,Teams Feedback Loop Integration
import requests
import mlflow

class TeamsFeedbackMonitor:
    """
    Monitor Teams channel for emoji reactions and log feedback to MLflow.
    """
    
    NEGATIVE_EMOJIS = ["👎", "❌", "😞", "😡", "😢", "😠", "dislike", "angry", "sad"]
    POSITIVE_EMOJIS = ["👍", "✅", "😊", "🎉", "❤️", "like", "love", "heart"]
    
    def __init__(self, graph_access_token, team_id, channel_id):
        """
        Initialize feedback monitor with Microsoft Graph API credentials.
        
        Args:
            graph_access_token: Microsoft Graph API access token
            team_id: The Teams team ID
            channel_id: The Teams channel ID
        """
        self.access_token = graph_access_token
        self.team_id = team_id
        self.channel_id = channel_id
        self.graph_api_base = "https://graph.microsoft.com/v1.0"
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        self.tracked_messages = {}
    
    def get_channel_messages(self, top=50):
        """
        Retrieve recent messages from the Teams channel.
        
        Args:
            top: Number of recent messages to retrieve
            
        Returns:
            List of message objects
        """
        url = f"{self.graph_api_base}/teams/{self.team_id}/channels/{self.channel_id}/messages"
        params = {"$top": top, "$expand": "replies"}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json().get("value", [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching messages: {e}")
            return []
    
    def get_message_reactions(self, message_id):
        """
        Get reactions for a specific message.
        
        Args:
            message_id: The message ID
            
        Returns:
            List of reaction objects
        """
        url = f"{self.graph_api_base}/teams/{self.team_id}/channels/{self.channel_id}/messages/{message_id}/reactions"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json().get("value", [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching reactions: {e}")
            return []
    
    def classify_sentiment(self, reactions):
        """
        Classify sentiment based on emoji reactions.
        
        Args:
            reactions: List of reaction objects
            
        Returns:
            Dict with sentiment classification and counts
        """
        negative_count = 0
        positive_count = 0
        
        for reaction in reactions:
            reaction_type = reaction.get("reactionType", "")
            
            if reaction_type in self.NEGATIVE_EMOJIS:
                negative_count += 1
            elif reaction_type in self.POSITIVE_EMOJIS:
                positive_count += 1
        
        total = negative_count + positive_count
        sentiment = "neutral"
        
        if total > 0:
            if negative_count > positive_count:
                sentiment = "negative"
            elif positive_count > negative_count:
                sentiment = "positive"
        
        return {
            "sentiment": sentiment,
            "negative_count": negative_count,
            "positive_count": positive_count,
            "total_reactions": total
        }
    
    def log_feedback_to_mlflow(self, message_content, query, response, sentiment_data, message_id):
        """
        Log feedback to MLflow for tracking and analysis.
        
        Args:
            message_content: The original message content
            query: The user query
            response: The agent response
            sentiment_data: Sentiment classification data
            message_id: The Teams message ID
        """
        try:
            with mlflow.start_run(run_name=f"feedback_{message_id[:8]}"):
                mlflow.log_param("message_id", message_id)
                mlflow.log_param("query", query)
                mlflow.log_param("sentiment", sentiment_data["sentiment"])
                
                mlflow.log_metric("negative_reactions", sentiment_data["negative_count"])
                mlflow.log_metric("positive_reactions", sentiment_data["positive_count"])
                mlflow.log_metric("total_reactions", sentiment_data["total_reactions"])
                
                mlflow.set_tag("feedback_source", "teams_reactions")
                mlflow.set_tag("requires_review", sentiment_data["sentiment"] == "negative")
                
                feedback_data = {
                    "message_id": message_id,
                    "query": query,
                    "response": response,
                    "sentiment": sentiment_data,
                    "timestamp": datetime.now().isoformat()
                }
                
                mlflow.log_dict(feedback_data, "feedback_details.json")
                
                print(f"✅ Logged feedback for message {message_id[:8]}: {sentiment_data['sentiment']}")
                
        except Exception as e:
            print(f"❌ Error logging feedback to MLflow: {e}")
    
    def track_message(self, message_id, query, response):
        """
        Start tracking a message for feedback.
        
        Args:
            message_id: The Teams message ID to track
            query: The user query
            response: The agent response
        """
        self.tracked_messages[message_id] = {
            "query": query,
            "response": response,
            "tracked_since": datetime.now()
        }
    
    def poll_feedback(self, log_to_mlflow=True):
        """
        Poll tracked messages for reactions and log feedback.
        
        Args:
            log_to_mlflow: Whether to log feedback to MLflow
            
        Returns:
            Dict with feedback summary
        """
        feedback_summary = {
            "total_checked": 0,
            "negative_feedback": 0,
            "positive_feedback": 0,
            "neutral_feedback": 0
        }
        
        messages = self.get_channel_messages()
        
        for message in messages:
            message_id = message.get("id")
            message_body = message.get("body", {}).get("content", "")
            
            if "🤖 AI Agent" in message_body or "AI Response" in message_body:
                feedback_summary["total_checked"] += 1
                
                reactions = self.get_message_reactions(message_id)
                sentiment_data = self.classify_sentiment(reactions)
                
                if message_id in self.tracked_messages:
                    tracked_info = self.tracked_messages[message_id]
                    query = tracked_info["query"]
                    response = tracked_info["response"]
                else:
                    query = "Unknown"
                    response = message_body
                
                if sentiment_data["sentiment"] == "negative":
                    feedback_summary["negative_feedback"] += 1
                    print(f"⚠️ Negative feedback detected on message {message_id[:8]}")
                    
                    if log_to_mlflow:
                        self.log_feedback_to_mlflow(
                            message_body, query, response, sentiment_data, message_id
                        )
                
                elif sentiment_data["sentiment"] == "positive":
                    feedback_summary["positive_feedback"] += 1
                    
                    if log_to_mlflow:
                        self.log_feedback_to_mlflow(
                            message_body, query, response, sentiment_data, message_id
                        )
                
                else:
                    feedback_summary["neutral_feedback"] += 1
        
        return feedback_summary

print("✅ Teams Feedback Monitor class defined")

# COMMAND ----------

# DBTITLE 1,Configure Teams Webhook
# IMPORTANT: Replace this with your actual Teams webhook URL
# To get a webhook URL:
# 1. Go to your Teams channel
# 2. Click the "..." menu → Connectors
# 3. Search for "Incoming Webhook"
# 4. Configure and copy the webhook URL

TEAMS_WEBHOOK_URL = "https://YOUR_ORGANIZATION.webhook.office.com/webhookb2/YOUR_WEBHOOK_ID"

# For demo purposes, we'll use a placeholder
# In production, use dbutils.secrets.get() to retrieve the webhook URL securely
# TEAMS_WEBHOOK_URL = dbutils.secrets.get(scope="teams", key="webhook_url")

print("⚠️  Note: Update TEAMS_WEBHOOK_URL with your actual Teams webhook URL")
print(f"Current webhook URL: {TEAMS_WEBHOOK_URL}")

# COMMAND ----------

# DBTITLE 1,Demo: Send Responses to Teams (Simulated)
# Since we may not have a real Teams webhook configured, we'll demonstrate the functionality

teams = TeamsIntegration(TEAMS_WEBHOOK_URL)

print("=" * 80)
print("TEAMS INTEGRATION DEMOS")
print("=" * 80)

# Demo 1: Simple message
print("\n1. Simple Message Format:")
print("-" * 40)
simple_message = {
    "title": "AI Agent Response",
    "message": "Hello from the AI Agent!"
}
print(json.dumps(simple_message, indent=2))
print("\n📤 Would send to Teams via webhook")

# Demo 2: Agent response with metadata
print("\n2. Agent Response with Metadata:")
print("-" * 40)
agent_response = {
    "title": "🤖 AI Agent Response",
    "sections": [
        {
            "title": "User Query",
            "text": "How do I restart my router?"
        },
        {
            "title": "Agent Response",
            "text": "To restart your ADSL-R500 router: 1) Unplug the power cable 2) Wait 30 seconds 3) Plug back in 4) Wait for all lights to turn green."
        },
        {
            "title": "Response Details",
            "facts": {
                "Model": "Claude Sonnet 4.5",
                "Response Time": "2.3s",
                "Tools Used": "product_documentation_retriever",
                "Confidence": "High"
            }
        }
    ]
}
print(json.dumps(agent_response, indent=2))
print("\n📤 Would send to Teams via webhook")

# Demo 3: RAG response with sources
print("\n3. RAG Response with Knowledge Sources:")
print("-" * 40)
rag_response = {
    "title": "🤖 AI Agent with RAG Response",
    "color": "00B294",
    "sections": [
        {
            "title": "📝 User Query",
            "text": "What is your refund policy?"
        },
        {
            "title": "💬 AI Response",
            "text": "You can cancel service anytime with 30 days notice. Refunds are prorated for unused service time. Equipment must be returned within 14 days."
        },
        {
            "title": "📚 Knowledge Sources Used",
            "sources": [
                "Policy Documentation: Service Cancellation and Refund Policy",
                "Policy Documentation: Billing Dispute Policy"
            ]
        }
    ]
}
print(json.dumps(rag_response, indent=2))
print("\n📤 Would send to Teams via webhook")

print("\n" + "=" * 80)
print("Note: To actually send to Teams, configure a real webhook URL")
print("=" * 80)

# COMMAND ----------

# DBTITLE 1,End-to-End Example: Query Agent → Send to Teams
def process_customer_query_and_notify(query, custom_prompt=None, send_to_teams=False):
    """
    Complete workflow: Query the agent and optionally send to Teams.
    
    Args:
        query: Customer query
        custom_prompt: Optional custom system prompt
        send_to_teams: Whether to send to Teams (requires valid webhook)
    """
    print(f"Processing query: {query}")
    print("-" * 80)
    
    start_time = time.time()
    
    try:
        # Invoke agent
        result = invoke_agent_with_custom_prompt(query, custom_prompt)
        
        end_time = time.time()
        latency = end_time - start_time
        
        # Extract response
        # Note: Adjust based on actual response format
        response_text = str(result)
        
        print(f"\n✅ Agent Response ({latency:.2f}s):")
        print(response_text)
        
        # Prepare metadata
        metadata = {
            "Model": "Claude Sonnet 4.5",
            "Response Time": f"{latency:.2f}s",
            "Endpoint": ENDPOINT_NAME,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Send to Teams if requested
        if send_to_teams and TEAMS_WEBHOOK_URL != "https://YOUR_ORGANIZATION.webhook.office.com/webhookb2/YOUR_WEBHOOK_ID":
            teams.send_agent_response(query, response_text, metadata)
            print("\n📤 Sent to Microsoft Teams")
        else:
            print("\n📋 Teams message prepared (not sent - configure webhook)")
            print(json.dumps({
                "query": query,
                "response": response_text,
                "metadata": metadata
            }, indent=2))
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        
        if send_to_teams and TEAMS_WEBHOOK_URL != "https://YOUR_ORGANIZATION.webhook.office.com/webhookb2/YOUR_WEBHOOK_ID":
            teams.send_error_notification(
                error_message=str(e),
                context={"Query": query, "Endpoint": ENDPOINT_NAME}
            )
            print("📤 Error notification sent to Teams")
        
        raise

# Test the end-to-end workflow
print("=" * 80)
print("END-TO-END WORKFLOW DEMO")
print("=" * 80)

queries = [
    "Give me information about john21@example.net",
    "How do I restart my ADSL-R500 router?",
    "What is your refund policy?",
]

for query in queries:
    print(f"\n{'=' * 80}")
    process_customer_query_and_notify(query, send_to_teams=False)
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Teams Feedback Loop - Monitor and Log User Reactions
# MAGIC
# MAGIC This section demonstrates how to monitor Teams messages for emoji reactions and automatically log feedback to MLflow.

# COMMAND ----------

# DBTITLE 1,Setup Feedback Monitor (Requires Microsoft Graph API Access)
GRAPH_ACCESS_TOKEN = "YOUR_GRAPH_API_ACCESS_TOKEN"
TEAM_ID = "YOUR_TEAM_ID"
CHANNEL_ID = "YOUR_CHANNEL_ID"

print("=" * 80)
print("TEAMS FEEDBACK MONITOR SETUP")
print("=" * 80)

print("""
To use the feedback monitor, you need:

1. Microsoft Graph API Access Token
   - Register an Azure AD app with permissions:
     * ChannelMessage.Read.All
     * ChannelMessageReaction.Read.All
   
2. Team ID and Channel ID
   - Get these from Teams URL or Graph API
   
3. Configure below with your credentials
""")

# COMMAND ----------

# DBTITLE 1,Initialize Feedback Monitor (Demo Mode)
feedback_monitor = None

if GRAPH_ACCESS_TOKEN != "YOUR_GRAPH_API_ACCESS_TOKEN":
    feedback_monitor = TeamsFeedbackMonitor(
        graph_access_token=GRAPH_ACCESS_TOKEN,
        team_id=TEAM_ID,
        channel_id=CHANNEL_ID
    )
    print("✅ Feedback monitor initialized")
else:
    print("⚠️ Feedback monitor not configured - using demo mode")
    feedback_monitor = None

# COMMAND ----------

# DBTITLE 1,Demo: Simulated Feedback Loop
print("=" * 80)
print("SIMULATED FEEDBACK LOOP DEMONSTRATION")
print("=" * 80)

simulated_messages = [
    {
        "message_id": "msg_001",
        "query": "What is your refund policy?",
        "response": "Our refund policy allows cancellations within 30 days...",
        "reactions": [
            {"reactionType": "like", "user": "user1"},
            {"reactionType": "like", "user": "user2"},
        ],
    },
    {
        "message_id": "msg_002",
        "query": "How do I fix Error Code 01?",
        "response": "I don't have specific information about Error Code 01.",
        "reactions": [
            {"reactionType": "dislike", "user": "user3"},
            {"reactionType": "dislike", "user": "user4"},
            {"reactionType": "angry", "user": "user5"},
        ],
    },
    {
        "message_id": "msg_003",
        "query": "Show me billing for john21@example.net",
        "response": "Here are the billing records: Bill 1: $89.99 - paid...",
        "reactions": [
            {"reactionType": "like", "user": "user6"},
        ],
    },
]

print("\n📊 Processing simulated feedback...")

if feedback_monitor:
    for msg in simulated_messages:
        feedback_monitor.track_message(msg["message_id"], msg["query"], msg["response"])

for msg in simulated_messages:
    print(f"\n{'─' * 80}")
    print(f"Message ID: {msg['message_id']}")
    print(f"Query: {msg['query']}")
    print(f"Response: {msg['response'][:60]}...")
    
    if feedback_monitor:
        sentiment_data = feedback_monitor.classify_sentiment(msg["reactions"])
    else:
        negative_count = sum(1 for r in msg["reactions"] if r["reactionType"] in TeamsFeedbackMonitor.NEGATIVE_EMOJIS)
        positive_count = sum(1 for r in msg["reactions"] if r["reactionType"] in TeamsFeedbackMonitor.POSITIVE_EMOJIS)
        total = negative_count + positive_count
        
        sentiment = "neutral"
        if total > 0:
            sentiment = "negative" if negative_count > positive_count else "positive"
        
        sentiment_data = {
            "sentiment": sentiment,
            "negative_count": negative_count,
            "positive_count": positive_count,
            "total_reactions": total
        }
    
    print(f"\n📈 Feedback Analysis:")
    print(f"   Sentiment: {sentiment_data['sentiment'].upper()}")
    print(f"   👍 Positive: {sentiment_data['positive_count']}")
    print(f"   👎 Negative: {sentiment_data['negative_count']}")
    print(f"   Total Reactions: {sentiment_data['total_reactions']}")
    
    if sentiment_data['sentiment'] == 'negative':
        print(f"\n⚠️ ALERT: Negative feedback detected! This should be reviewed.")
        print(f"   Action: Logging to MLflow for review and analysis")
        
        with mlflow.start_run(run_name=f"negative_feedback_{msg['message_id']}"):
            mlflow.log_param("message_id", msg['message_id'])
            mlflow.log_param("query", msg['query'])
            mlflow.log_param("sentiment", sentiment_data['sentiment'])
            mlflow.log_metric("negative_reactions", sentiment_data['negative_count'])
            mlflow.log_metric("positive_reactions", sentiment_data['positive_count'])
            mlflow.set_tag("requires_review", "true")
            mlflow.set_tag("feedback_source", "teams_reactions")
        
        print(f"   ✅ Logged to MLflow experiment")

print(f"\n{'=' * 80}")
print("FEEDBACK SUMMARY")
print("=" * 80)
print(f"Total Messages Analyzed: {len(simulated_messages)}")
negative_msgs = sum(1 for msg in simulated_messages if 
                    sum(1 for r in msg['reactions'] if r['reactionType'] in TeamsFeedbackMonitor.NEGATIVE_EMOJIS) >
                    sum(1 for r in msg['reactions'] if r['reactionType'] in TeamsFeedbackMonitor.POSITIVE_EMOJIS))
print(f"Messages with Negative Feedback: {negative_msgs}")
print(f"Feedback logged to MLflow for review and agent improvement")

# COMMAND ----------

# DBTITLE 1,Automated Feedback Polling (Production Pattern)
print("=" * 80)
print("PRODUCTION FEEDBACK POLLING PATTERN")
print("=" * 80)

print("""
For production deployment, set up scheduled polling:

1. Use Databricks Jobs to run feedback polling on a schedule (e.g., hourly)

2. Example polling code:
   
   def scheduled_feedback_poll():
       feedback_monitor = TeamsFeedbackMonitor(
           graph_access_token=get_access_token(),
           team_id=TEAM_ID,
           channel_id=CHANNEL_ID
       )
       
       summary = feedback_monitor.poll_feedback(log_to_mlflow=True)
       
       if summary['negative_feedback'] > 0:
           send_alert_to_slack(f"Found {summary['negative_feedback']} negative feedback items")
       
       return summary

3. Create alerts for negative feedback thresholds

4. Use MLflow experiments to track feedback trends over time

5. Integrate feedback into agent retraining pipeline

Benefits:
✅ Automatic detection of poor responses
✅ Continuous improvement through user feedback
✅ Early warning system for agent quality issues
✅ Data-driven agent refinement
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Summary and Key Takeaways

# COMMAND ----------

# DBTITLE 1,Demo Summary
print("=" * 80)
print("🎉 DEMO COMPLETE - Summary of What We Built")
print("=" * 80)

summary = f"""
✅ 1. LLM Setup
   - Model: Claude Sonnet 4.5 via Databricks Model Serving
   - Endpoint: {LLM_ENDPOINT_NAME}

✅ 2. Data & Tools
   - Created 3 sample data tables (customers, billing, support_tickets)
   - Created 3 Unity Catalog SQL Functions as agent tools
   - Tools accessible via: {catalog}.{schema}.*

✅ 3. Multiple Knowledge Bases (RAG)
   - Product Documentation Index: {product_docs_index}
   - Troubleshooting Guide Index: {troubleshooting_index}  
   - Policy Documentation Index with Confluence metadata: {policy_index}
     * Includes source URLs for citation
     * HTML content from Confluence API
     * Tags and metadata for traceability
     * Custom Citation-Formatting Retriever embeds URLs in text
   - Vector Search Endpoint: {VECTOR_SEARCH_ENDPOINT_NAME}

✅ 4. Prompt Engineering
   - Tested 3 different system prompt versions
   - Version 3 (optimized) selected for production
   - Demonstrated impact on response quality

✅ 5. Agent Creation
   - Built LangGraph-based agent with multiple retrievers
   - Integrated 3 UC functions + 3 vector search indexes
   - Total of 6 tools available to agent

✅ 6. Agent Evaluation with MLflow
   - Created synthetic evaluation dataset with AI_QUERY
   - Defined custom scorers: Groundedness, Relevance, Safety, Guidelines
   - Ran evaluations on multiple agent versions
   - Compared prompt variations to measure impact on quality
   - Dataset: {eval_dataset_table_name}

✅ 7. MLflow Integration
   - Logged agent with full tracing
   - Registered to Unity Catalog: {UC_MODEL_NAME}
   - Version: {uc_registered_model_info.version}

✅ 8. Model Serving Deployment
   - Deployed to endpoint: {ENDPOINT_NAME}
   - Endpoint supports dynamic system prompts
   - Scalable, production-ready serving

✅ 9. Dynamic System Prompts
   - Demonstrated 5 different prompt variations:
     * Default professional tone
     * Friendly/casual tone
     * Technical/detailed responses
     * Brief/concise responses
     * Multilingual (Spanish) support
   - Prompts include citation requirements for policy information

✅ 10. Microsoft Teams Integration
   - Created Teams integration class
   - Supports formatted messages, errors, and RAG responses
   - End-to-end workflow demonstrated

✅ 11. Teams Feedback Loop
   - Automated feedback monitoring via Microsoft Graph API
   - Polls channel for emoji reactions on agent messages
   - Detects negative feedback (👎, ❌, 😞, 😡, etc.)
   - Automatically logs feedback to MLflow for tracking
   - Production pattern with scheduled polling demonstrated

KEY DATABRICKS FEATURES SHOWCASED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Unity Catalog - for data, functions, and model governance
• Vector Search - for semantic search and RAG
• Model Serving - for scalable agent deployment  
• MLflow 3.0 - for experiment tracking, tracing, and evaluation
• MLflow Evaluation - with custom scorers and guidelines
• MLflow Feedback Logging - for user reaction tracking
• AI_QUERY - for synthetic evaluation dataset generation
• LangChain/LangGraph - for agent orchestration
• Agent Framework - for responses protocol and streaming
• Databricks Notebooks - for interactive development
• Databricks Jobs - for scheduled feedback polling (production pattern)
"""

print(summary)

displayHTML(f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 30px; 
            border-radius: 10px; 
            color: white; 
            font-family: Arial, sans-serif;">
    <h2 style="margin-top: 0;">🎯 Next Steps</h2>
    <ul style="line-height: 2;">
        <li>✨ Customize the system prompts for your use case</li>
        <li>📚 Add more knowledge bases for additional domains</li>
        <li>🔗 Connect to real Confluence API for live document sync</li>
        <li>🔧 Create more UC functions for specific business logic</li>
        <li>📊 Expand evaluation dataset with real customer queries</li>
        <li>🚀 Configure Teams webhook and Graph API for production</li>
        <li>📡 Set up scheduled feedback polling with Databricks Jobs</li>
        <li>🔒 Implement access controls and audit logging</li>
        <li>📈 Build feedback dashboards to track agent quality trends</li>
        <li>📝 Add citation tracking for compliance reporting</li>
    </ul>
    <p style="margin-bottom: 0;">
        <strong>Resources:</strong><br>
        <a href="/ml/endpoints/{ENDPOINT_NAME}" target="_blank" style="color: #ffd700;">View Model Serving Endpoint</a> | 
        <a href="/explore/data/models/{catalog}/{schema}/{MODEL_NAME}" target="_blank" style="color: #ffd700;">View Model in Unity Catalog</a>
    </p>
</div>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix: Utility Functions

# COMMAND ----------

# DBTITLE 1,Cleanup Function (Optional)
def cleanup_demo_resources(confirm=False):
    """
    Clean up all demo resources. Use with caution!
    
    Args:
        confirm: Must be True to actually perform cleanup
    """
    if not confirm:
        print("⚠️  This function will delete all demo resources.")
        print("To confirm, call: cleanup_demo_resources(confirm=True)")
        return
    
    print("🗑️  Cleaning up demo resources...")
    
    # Delete vector search indexes
    try:
        vsc.delete_index(VECTOR_SEARCH_ENDPOINT_NAME, product_docs_index)
        print(f"✅ Deleted index: {product_docs_index}")
    except:
        pass
    
    try:
        vsc.delete_index(VECTOR_SEARCH_ENDPOINT_NAME, troubleshooting_index)
        print(f"✅ Deleted index: {troubleshooting_index}")
    except:
        pass
    
    try:
        vsc.delete_index(VECTOR_SEARCH_ENDPOINT_NAME, policy_index)
        print(f"✅ Deleted index: {policy_index}")
    except:
        pass
    
    # Delete tables
    tables = ["customers", "billing", "support_tickets", "product_docs", "troubleshooting_docs", "policy_docs"]
    for table in tables:
        try:
            spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}.{table}")
            print(f"✅ Deleted table: {table}")
        except:
            pass
    
    # Delete UC functions
    functions = ["get_customer_info", "get_billing_info", "get_support_tickets"]
    for func in functions:
        try:
            spark.sql(f"DROP FUNCTION IF EXISTS {catalog}.{schema}.{func}")
            print(f"✅ Deleted function: {func}")
        except:
            pass
    
    print("✅ Cleanup complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **End of Demo Notebook**
# MAGIC
# MAGIC This notebook demonstrated a complete end-to-end AI Agent implementation using:
# MAGIC - Claude Sonnet 4.5 as the foundation model
# MAGIC - Multiple Vector Search indexes for RAG
# MAGIC - Unity Catalog SQL Functions as agent tools
# MAGIC - Confluence-style document integration with source citations
# MAGIC - MLflow for evaluation and deployment
# MAGIC - Dynamic system prompts for customizable behavior
# MAGIC - Microsoft Teams integration with feedback loop
# MAGIC
# MAGIC **Key Features:**
# MAGIC - Retrieval Augmented Generation with multiple knowledge bases
# MAGIC - Source attribution and citations
# MAGIC - Agent evaluation with custom metrics
# MAGIC - Real-time model serving
# MAGIC - User feedback monitoring
# MAGIC
# MAGIC All powered by the Databricks AI Agent Framework! 🚀
# MAGIC
