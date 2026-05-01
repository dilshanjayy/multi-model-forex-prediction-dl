import { create } from 'zustand';

export const useStore = create((set) => ({
  activeTab: 'terminal',
  setActiveTab: (tab) => set({ activeTab: tab }),

  models: [],
  setModels: (models) => set({ models }),
  
  selectedModel: null,
  setSelectedModel: (model) => set({ selectedModel: model }),

  modelDetails: null,
  setModelDetails: (details) => set({ modelDetails: details }),

  liveData: null,
  setLiveData: (data) => set({ liveData: data }),

  lotSize: 0.1,
  setLotSize: (size) => set({ lotSize: size }),

  token: localStorage.getItem('token') || null,
  setToken: (token) => {
    if (token) {
      localStorage.setItem('token', token);
    } else {
      localStorage.removeItem('token');
    }
    set({ token });
  },

  user: null,
  setUser: (user) => set({ user }),
  
  logout: () => {
    localStorage.removeItem('token');
    set({ token: null, user: null });
  },
}));
