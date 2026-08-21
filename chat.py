import asyncio
import os
import sys
import uuid
from dotenv import load_dotenv

# Load environment variables (.env file)
load_dotenv()

# Windows event loop policy patch for psycopg async connection
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from src.core.graph import get_compiled_graph
from src.db.session import get_checkpointer
from src.core.state import ProjectState

async def chat_loop():
    print("====================================================")
    print("🤖 Welcome to ResearchRealy Interactive Chat Advisor!")
    print("Type your prompts below to converse with the orchestrator.")
    print("Type 'exit' or 'quit' to end the session.")
    print("====================================================")

    thread_id = f"chat_{uuid.uuid4().hex[:6]}"
    print(f"[*] Conversation Thread ID: {thread_id}\n")

    async with get_checkpointer() as checkpointer:
        app = get_compiled_graph(checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        
        state_values = {
            "project_id": f"proj_{thread_id}",
            "project_root": os.getcwd(),
            "tech_stack": {},
            "locked_decisions": [],
            "task_backlog": [],
            "active_task_id": None,
            "retrieved_code_context": [],
            "retrieved_web_docs": [],
            "generated_prompt_payload": {},
            "critic_iteration": 0,
            "critic_passed": False,
            "critic_feedback": None,
            "permission_granted": False,
            "conversational_response": None
        }

        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit"]:
                    print("\n[*] Exiting chat. Goodbye!")
                    break

                print("\n[Thinking...]")
                
                # Restore current state from database checkpointer
                current_state = await app.aget_state(config)
                if current_state.values:
                    state_values = current_state.values

                # Detect if we were waiting for permission and user confirmed
                was_waiting = state_values.get("conversational_response") is not None
                is_confirmation = user_input.lower() in ["yes", "y", "proceed", "go ahead", "do it", "sure", "ok"]

                if was_waiting and is_confirmation:
                    state_values["permission_granted"] = True
                    state_values["conversational_response"] = None
                    # Objective remains unchanged (carries over previous task)
                else:
                    # Brand new user prompt/objective (requires permission evaluation)
                    state_values["permission_granted"] = False
                    state_values["conversational_response"] = None
                    payload = dict(state_values.get("generated_prompt_payload") or {})
                    payload["objective"] = user_input
                    state_values["generated_prompt_payload"] = payload

                # Reset validation iterations for the new prompt run
                state_values["critic_iteration"] = 0
                state_values["critic_passed"] = False
                state_values["critic_feedback"] = None

                # Invoke the unified LangGraph pipeline
                result = await app.ainvoke(state_values, config)
                
                # Output Results
                print("\n🤖 Advisor:")
                conv_resp = result.get("conversational_response")
                
                if conv_resp:
                    # Print permission prompt
                    print(conv_resp)
                else:
                    # Permission granted or standard intent execution complete
                    payload = result.get("generated_prompt_payload") or {}
                    intent = payload.get("intent", "DECOMPOSE")
                    backlog = result.get("task_backlog", [])
                    
                    if intent == "DECOMPOSE":
                        print("[+] Project decomposition completed. Generated tasks:")
                        for idx, t in enumerate(backlog):
                            status = t.status if hasattr(t, "status") else t.get("status", "PENDING")
                            t_id = t.task_id if hasattr(t, "task_id") else t.get("task_id", "")
                            title = t.title if hasattr(t, "title") else t.get("title", "")
                            print(f"  {idx+1}. [{status}] {t_id}: {title}")
                    
                    elif intent == "CODE":
                        passed = result.get("critic_passed", False)
                        feedback = result.get("critic_feedback")
                        if passed:
                            print("[+] Specifications generated and validated successfully!")
                            for t in backlog:
                                t_id = t.task_id if hasattr(t, "task_id") else t.get("task_id", "")
                                if t_id == result.get("active_task_id"):
                                    files = t.target_files if hasattr(t, "target_files") else t.get("target_files", [])
                                    print(f"📄 Approved blueprint written to: {files}")
                        else:
                            print(f"[-] Validation failed: {feedback}")
                    
                    else:
                        print("[+] Goal processed successfully.")
                        
            except KeyboardInterrupt:
                print("\n[*] Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n[!] Error: {e}")

if __name__ == "__main__":
    asyncio.run(chat_loop())
