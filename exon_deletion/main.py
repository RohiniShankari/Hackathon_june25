# main.py
import os, boto3, re
from langchain import hub
# from langchain_core.tools import tool, Tool, StructuredTool
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain_aws import ChatBedrock
from tools.main_tools import retrieve_tools

tools = retrieve_tools()

def extract_paths_and_clean_output(text):
    # Extract full file paths (e.g., /home/ubuntu/...)
    paths = re.findall(r"/[\w./\-]+(?:\.vcf|\.csv|\.fasta|\.png)", text)

    return text, list(set(paths))

def run_agent(input_prompt):
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    llm = ChatBedrock(
        model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
        model_kwargs={"temperature": 0},
        client=client,
    )

    prompt = hub.pull("hwchase17/structured-chat-agent")

    agent = create_structured_chat_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    result = agent_executor.invoke({
        "input": input_prompt
    })

    output_text, extracted_paths = extract_paths_and_clean_output(result["output"])

    print("Extracted Paths:", extracted_paths)
    for path in extracted_paths:
        output_text = output_text.replace(path, "attached below")

    return output_text, extracted_paths
    # png_files = []
    # csv_files = []
    # other_files = []
    # for line in output_text.splitlines():
    #     if ".png" in line:
    #         png_files.append(line.strip())
    #     elif ".csv" in line:
    #         csv_files.append(line.strip())
    #     elif ".vcf" in line:
    #         other_files.append(line.strip())

    # response =  {
    #     "output": output_text,
    #     "png": png_files,
    #     "csv": csv_files,
    #     "other": other_files
    # }
    # print(response)

    # return response

if __name__=="__main__":
    # prompt = "Delete exon from given sequence AAGTGTCTTTGCAGCTGTGGTGGCTCAGAGCAGGTCAGAGGCTCTGCTGTCTGTGTAGTGAGTGCAGTTGCCTTGAGTGACTCAGGGAAGAGGTGTAGTGAGGAAACAGGGGAGATCAGGTGTTTTCATGTTTGTGTGTTTGTTTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTTTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCTGCTGTCCTGCTGTTTGTTGCTGTGTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTCTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTTGTTTT at positions 200 and 205"
    # prompt = "Delete exon from given fasta file at /home/ubuntu/exon_deletion/inputs/tp53.fasta and positions 200 and 205"
    # prompt = "Get exon coordinates of TP53 gene and predict splicing changes if exon at index 1 on chromosome 17 is deleted."
    prompt = "Get exon coordinates of TP53 gene"
    # prompt = "Predict splicing changes if exon at positions 7676521 and 7676594 on chromosome 17 is deleted."
    # prompt = "Compare regulatory changes between WT and deleted exon sequence using Enformer. WT file is in '/home/ubuntu/exon_deletion/outputs/api/deleted_exon_WT.fasta', delta file is in '/home/ubuntu/exon_deletion/outputs/api/deleted_exon_DEL.fasta', and the exon spans 15000 to 15150."
    # prompt = "Compare regulatory changes using wt_file and delta_file"
    run_agent(prompt)
